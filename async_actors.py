"""Independent DCSS actors with centralized batched neural inference."""
from __future__ import annotations

import queue
import random
import threading
import time
from dataclasses import dataclass

import torch

from dcss_env import DCSSEnv
from r2d2 import RecurrentQ, Transition
from train_rl import encode


@dataclass
class InferenceRequest:
    actor: int
    observation: str
    action_mask: list[bool]
    previous_action: int
    reset: bool


@dataclass
class InferenceResponse:
    action: int
    probabilities: list[float]
    value: float


@dataclass
class ActorEvent:
    actor: int
    transition: Transition
    info: dict
    screen: str
    colors: str
    probabilities: list[float]


class BatchedInferenceServer:
    def __init__(self, model: RecurrentQ, actor_count: int, device,
                 batch_size=32, batch_wait_ms=2.0, epsilons=None):
        self.device = device
        self.model = RecurrentQ(model.n_actions).to(device).eval()
        self.model.load_state_dict(model.state_dict())
        self.actor_count = actor_count
        self.batch_size = batch_size
        self.batch_wait = batch_wait_ms / 1000
        self.epsilons = epsilons or [0.01] * actor_count
        self.requests = queue.Queue()
        self.responses = [queue.Queue(maxsize=1) for _ in range(actor_count)]
        self.hidden = self.model.initial_state(actor_count, device)
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.thread = threading.Thread(
            target=self._run, name="batched-inference", daemon=True)

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.requests.put(None)
        self.thread.join(timeout=10)

    def sync_from(self, model):
        state = {name: value.detach().cpu().clone()
                 for name, value in model.state_dict().items()}
        with self.lock:
            self.model.load_state_dict(state)

    def infer(self, request):
        self.requests.put(request)
        return self.responses[request.actor].get()

    def _run(self):
        while not self.stop_event.is_set():
            try:
                first = self.requests.get(timeout=0.5)
            except queue.Empty:
                continue
            if first is None:
                continue
            batch = [first]
            deadline = time.perf_counter() + self.batch_wait
            while len(batch) < self.batch_size:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                try:
                    item = self.requests.get(timeout=remaining)
                except queue.Empty:
                    break
                if item is not None:
                    batch.append(item)
            actors = [request.actor for request in batch]
            for request in batch:
                if request.reset:
                    self.hidden[request.actor].zero_()
            observations = torch.stack([
                encode(request.observation) for request in batch]).to(self.device)
            masks = torch.tensor(
                [request.action_mask for request in batch],
                dtype=torch.bool, device=self.device)
            previous = torch.tensor(
                [request.previous_action for request in batch],
                dtype=torch.long, device=self.device)
            batch_hidden = self.hidden[actors]
            with self.lock, torch.no_grad():
                q, next_hidden = self.model.step(
                    observations, previous, batch_hidden, masks)
                probabilities = torch.softmax(q, dim=-1)
            self.hidden[actors] = next_hidden
            for row, request in enumerate(batch):
                legal = [index for index, allowed
                         in enumerate(request.action_mask) if allowed]
                if random.random() < self.epsilons[request.actor]:
                    action = random.choice(legal)
                else:
                    action = int(q[row].argmax())
                self.responses[request.actor].put(InferenceResponse(
                    action,
                    [float(value) for value in probabilities[row].cpu()],
                    float(q[row].max().cpu())))


class AsyncActorPool:
    def __init__(self, count, variant, max_steps, target_depth,
                 inference: BatchedInferenceServer):
        self.count = count
        self.variant = variant
        self.max_steps = max_steps
        self.target_depth = target_depth
        self.inference = inference
        self.events = queue.Queue(maxsize=max(128, count * 8))
        self.stop_event = threading.Event()
        self.threads = [threading.Thread(
            target=self._actor, args=(index,), name=f"crawl-actor-{index}",
            daemon=True) for index in range(count)]

    def start(self):
        self.inference.start()
        for thread in self.threads:
            thread.start()

    def stop(self):
        self.stop_event.set()
        deadline = time.time() + 30
        while time.time() < deadline and any(
                thread.is_alive() for thread in self.threads):
            try:
                self.events.get_nowait()
            except queue.Empty:
                pass
            for thread in self.threads:
                thread.join(timeout=0.05)
        self.inference.stop()

    def _actor(self, actor):
        env = DCSSEnv(
            actor, target_depth=self.target_depth,
            max_steps=self.max_steps, variant=self.variant)
        previous_action = -1
        reset = True
        try:
            observation = env.reset()
            while not self.stop_event.is_set():
                action_mask = env.action_mask()
                response = self.inference.infer(InferenceRequest(
                    actor, observation, action_mask, previous_action, reset))
                reset = False
                colors = env.color_text()
                next_observation, reward, done, info = env.step(response.action)
                next_mask = info.get("action_mask", env.action_mask())
                transition = Transition(
                    encode(observation).to(torch.uint8), previous_action,
                    response.action, float(reward), bool(done),
                    torch.tensor(action_mask, dtype=torch.bool),
                    encode(next_observation).to(torch.uint8),
                    torch.tensor(next_mask, dtype=torch.bool))
                self.events.put(ActorEvent(
                    actor, transition, info, observation, colors,
                    response.probabilities))
                previous_action = response.action
                observation = next_observation
                if done:
                    observation = env.reset()
                    previous_action = -1
                    reset = True
        finally:
            env.close()
