"""
PPO on DCSS. The policy starts from random weights and learns from its own
games. No teacher, no demonstrations, no imitation term anywhere in this file.

    /root/pty-venv/bin/python train_rl.py --envs 8 --updates 400

Design notes that matter for reproducing this
---------------------------------------------
* Observation is the raw 80x24 terminal screen as character ids, plus one
  player-visible bit on map cells occupied by a hostile. It never receives
  hidden state; the extra bit disambiguates monster glyphs from scenery in the
  same way the visible monster list does for a human.
* Actions are DCSS macro commands (see dcss_env.ACTIONS), which is what makes
  an episode ~200 decisions instead of ~10,000 keystrokes.
* Rollouts are collected in threads because env.step is pure blocking I/O on a
  pty; the GIL is released in select()/read(), so this genuinely parallelises.
* Envs auto-reset mid-rollout and the value function bootstraps across the
  truncation, so a fixed-length rollout does not bias returns.
"""
import argparse
import json
import random
import time
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from dcss_env import VARIANTS, DCSSEnv

HERE = Path(__file__).parent
DATA = HERE / "data"
GAMES = HERE / "games.jsonl"

# Every artefact is per-variant so three runs can share the machine without
# overwriting each other's logs, checkpoints or live feeds.
LOG = CKPT = REPLAYS = LIVE = ENVS = VIEW = None


def _mean(vals):
    """Mean of the readable values only. None means "the status panel was
    covered this instant", which is not a zero — see the mean_ac note below."""
    seen = [v for v in vals if v is not None]
    return round(sum(seen) / len(seen), 2) if seen else None


def set_paths(v):
    global LOG, CKPT, REPLAYS, LIVE, ENVS, VIEW
    LOG = DATA / f"rl_log.{v}.jsonl"
    CKPT = DATA / f"rl_policy.{v}.pt"
    REPLAYS = DATA / f"rl_replays_{v}"
    LIVE = DATA / f"rl_live.{v}.json"
    ENVS = DATA / f"rl_envs.{v}.json"
    VIEW = DATA / f"rl_view.{v}.txt"

BASE_VOCAB = 128     # ASCII; anything else is folded to '?'
VOCAB = 256          # second half = same glyph with visible-hostile bit set
COLS, ROWS = 80, 24
SCREEN_CHARS = COLS * ROWS
CROP = 15            # egocentric window, odd so the player sits dead centre

# --- model size: kept small on purpose. The bottleneck is the game (~12
# steps/s), not the matmul, so a bigger net buys nothing but slower updates.
D_MODEL = 64
N_LAYER = 2
N_HEAD = 4
POOL = 8             # wide map: 1920 chars -> 240 positions (crop is unpooled)


def to_grid(screen_text):
    """Screen text -> LongTensor[24, 80] of character ids, properly aligned.

    The previous encoder flattened `"\\n".join(rows)` into one 1920-char string.
    That was wrong three ways, all measured:
      * rows are 81 chars once the newline is counted, so every 16-char pooling
        block straddled a row boundary, and the straddle drifted down the screen
        (row 1 offset 1, row 2 offset 2, ... row 23 offset 7);
      * the joined text is 1943 chars, so truncating at 1920 silently cut the
        last 23 characters — the tail of the bottom message line;
      * a 2-D dungeon was treated as a 1-D string, putting the cell directly
        above the player 81 positions away instead of adjacent.
    Splitting on newlines and padding each row to exactly 80 fixes all three.
    """
    rows = screen_text.split("\n")[:ROWS]
    out = []
    for r in range(ROWS):
        line = rows[r] if r < len(rows) else ""
        b = line.encode("ascii", "replace")[:COLS]
        ids = list(b) + [32] * (COLS - len(b))
        out.append([c if c < BASE_VOCAB else 63 for c in ids])
    return torch.tensor(out, dtype=torch.long)


AT = ord("@")


def visible_hostile_cells(screen_text):
    """Conservative hostile overlay inferred from the visible terminal map.

    In console Crawl, hostile positions are letter glyphs in the left dungeon
    pane.  Menus are excluded by requiring a map-like amount of terrain.  The
    WebTiles adapter supplies exact player-visible ``mon.att`` positions
    instead, including monsters whose glyph is punctuation.
    """
    if sum(screen_text.count(c) for c in "#.") <= 60:
        return set()
    grid = screen_text.split("\n")
    return {(x, y) for y, row in enumerate(grid[:17])
            for x, char in enumerate(row[:37])
            if char != "@" and char.isascii() and char.isalpha()}


def encode(screen_text, hostile_cells=None):
    """-> LongTensor[CROP*CROP + ROWS*COLS]: an egocentric crop then the map.

    The crop is the important half. At full resolution, centred on the player,
    the network can finally see that a monster is ADJACENT rather than
    somewhere in a 16-character average. The full grid follows so it still has
    the global shape needed to navigate toward stairs.
    """
    g = to_grid(screen_text)
    cells = (visible_hostile_cells(screen_text) if hostile_cells is None
             else hostile_cells)
    for x, y in cells:
        if 0 <= x < COLS and 0 <= y < ROWS and g[y, x] < BASE_VOCAB:
            g[y, x] += BASE_VOCAB
    # Find the player. Search only the map region: '@' also appears in prose
    # ("@ the Chopper") and in the status panel.
    region = g[:17, :37]
    hit = (region == AT).nonzero()
    if len(hit):
        py, px = int(hit[0][0]), int(hit[0][1])
    else:
        py, px = ROWS // 2, COLS // 4      # no player visible (menu, death)

    half = CROP // 2
    pad = torch.full((ROWS + CROP, COLS + CROP), 32, dtype=torch.long)
    pad[half:half + ROWS, half:half + COLS] = g
    crop = pad[py:py + CROP, px:px + CROP]
    return torch.cat([crop.reshape(-1), g.reshape(-1)])


OBS_LEN = CROP * CROP + SCREEN_CHARS


def apply_action_mask(logits, mask):
    """Remove impossible UI actions without changing scores among legal ones.

    ``mask`` is part of the observation contract, not a reward trick.  PPO must
    use the same mask when collecting *and* optimizing a transition; otherwise
    the stored log probability and the updated distribution describe different
    action spaces.
    """
    if mask is None:
        return logits
    mask = mask.to(device=logits.device, dtype=torch.bool)
    if mask.ndim == 1:
        mask = mask.unsqueeze(0)
    if not torch.all(mask.any(dim=-1)):
        raise ValueError("action mask contains a state with no legal action")
    return logits.masked_fill(~mask, torch.finfo(logits.dtype).min)


class Policy(nn.Module):
    """Shared trunk, separate actor and critic heads."""

    def __init__(self, n_actions=6):
        super().__init__()
        self.n_actions = n_actions
        self.emb = nn.Embedding(VOCAB, D_MODEL)
        # The crop is NOT pooled: every one of its 225 cells keeps its own
        # embedding, so an adjacent monster is a distinct token rather than
        # 1/16th of an average. Only the wider map is downsampled, and by 8
        # rather than 16 — it is context, not the thing being reacted to.
        self.pool = nn.AvgPool1d(POOL)
        n_pos = CROP * CROP + SCREEN_CHARS // POOL
        self.pos = nn.Parameter(torch.zeros(1, n_pos, D_MODEL))
        layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=N_HEAD, dim_feedforward=D_MODEL * 4,
            dropout=0.0, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=N_LAYER)
        self.actor = nn.Linear(D_MODEL, n_actions)
        self.critic = nn.Linear(D_MODEL, 1)
        # Small init on the actor keeps the initial policy near-uniform. A
        # confidently wrong start collapses entropy before any reward is seen.
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.zeros_(self.actor.bias)

    def _tokens(self, x):
        """x: B, CROP*CROP + SCREEN_CHARS  ->  B, n_pos, D"""
        n = CROP * CROP
        crop = self.emb(x[:, :n])                       # B, 225, D  full res
        wide = self.emb(x[:, n:])                       # B, 1920, D
        wide = self.pool(wide.transpose(1, 2)).transpose(1, 2)
        return torch.cat([crop, wide], 1) + self.pos

    def forward(self, x):
        h = self.enc(self._tokens(x)).mean(1)
        return self.actor(h), self.critic(h).squeeze(-1)

    def act(self, x, action_mask=None):
        logits, v = self(x)
        dist = torch.distributions.Categorical(
            logits=apply_action_mask(logits, action_mask))
        a = dist.sample()
        return a, dist.log_prob(a), dist.entropy(), v

    def explain(self, x, action_mask=None):
        """What the network computed for this screen, for the replay viewer.

        Returns per-action probabilities, the value estimate, and the encoder's
        activation magnitude at each of the 120 pooled positions. Position p
        covers screen characters [p*POOL, (p+1)*POOL), so the map lines up with
        rows of the 80x24 grid.

        This is activation MAGNITUDE, not attribution: it shows where the
        encoder is carrying signal, not a causal claim that the model decided
        because of those cells. Labelled that way in the UI on purpose.
        """
        h = self.enc(self._tokens(x))         # B, P, D
        pooled = h.mean(1)
        logits = apply_action_mask(self.actor(pooled), action_mask)
        v = self.critic(pooled).squeeze(-1)
        sal = h.norm(dim=-1)                  # B, P
        sal = sal / sal.amax(dim=1, keepdim=True).clamp(min=1e-6)
        return torch.softmax(logits, -1), v, sal


def load_policy_state(policy, state_dict):
    """Load an older policy while preserving compatible learned behaviour.

    Action rows are semantic and append-only: an old actor is copied into the
    leading rows exactly.  The expanded hostile vocabulary begins as a copy of
    the corresponding plain glyph embedding, so migration is behaviour-neutral
    until PPO learns to use the new visible bit.
    """
    own = policy.state_dict()
    loaded, expanded, skipped = [], [], []
    for name, source in state_dict.items():
        if name not in own:
            skipped.append(name)
            continue
        target = own[name]
        if source.shape == target.shape:
            target.copy_(source)
            loaded.append(name)
        elif name in {"actor.weight", "actor.bias"} and source.shape[0] <= target.shape[0]:
            target[:source.shape[0]].copy_(source)
            # New actions start deliberately modest, but remain sampleable.
            if name == "actor.bias":
                target[source.shape[0]:].fill_(float(source.min()) - 1.5)
            expanded.append(name)
        elif name == "emb.weight" and source.shape[1:] == target.shape[1:] and source.shape[0] <= target.shape[0]:
            target[:source.shape[0]].copy_(source)
            remaining = target.shape[0] - source.shape[0]
            if remaining:
                target[source.shape[0]:].copy_(source[:remaining])
            expanded.append(name)
        else:
            skipped.append(name)
    policy.load_state_dict(own)
    return {"loaded": loaded, "expanded": expanded, "skipped": skipped}


class Recorder:
    """Save watchable episodes so the RL agent shows up in the dashboard.

    One file per episode under data/rl_replays/, which the panel's existing
    replay modal reads directly — the screen is already part of every
    transition, so nothing extra needs to be captured and no terminal emulator
    is needed in the browser.

    Not every episode is kept: at ~30 steps/s a full episode is roughly 1MB of
    screens. Keeping env 0 gives a steady sample of typical play, and keeping
    every SOLVE means the interesting games are never the ones thrown away.
    """

    def __init__(self, enabled=True, sample_env=0, keep=40, names=(), variant="a"):
        self.enabled = enabled
        self.sample_env = sample_env
        self.keep = keep
        # The viewer needs the ORDERED action names to label the probability
        # vector. Variants have 7, 10 and 14 actions, so a hardcoded list in
        # the dashboard mislabels everything past the sixth entry.
        self.names = list(names)
        self.variant = variant
        self.buf = {}
        self._watch = 0
        self._watch_t = 0.0
        if enabled:
            REPLAYS.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _write(path, obj):
        """Write via temp+rename so a reader never sees a half-written file."""
        try:
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(obj), encoding="utf-8")
            tmp.replace(path)
        except OSError:
            pass

    def watched(self):
        """Which env the dashboard is currently showing. Re-read at most once a
        second — this is called on every env step, and hitting /mnt/c that often
        is not free."""
        now = time.time()
        if now - self._watch_t > 1.0:
            self._watch_t = now
            try:
                self._watch = int(VIEW.read_text().strip())
            except (OSError, ValueError):
                self._watch = 0
        return self._watch

    def publish_envs(self, envs):
        """One-line status for every game, so all N can be seen at a glance."""
        self._write(ENVS, [{
            "env": i, "depth": e.max_depth, "xl": e.xl, "turns": e.turns,
            "hp": round(e.hp_frac, 2), "steps": e.steps,
            "outcome": e.outcome,
        } for i, e in enumerate(envs)])

    def step(self, i, screen, colors, action, t, probs=None, value=None, sal=None):
        if not self.enabled:
            return
        f = {"t": round(t, 2), "action": action, "state": screen,
             "colors": colors}
        if probs is not None:
            f["probs"] = [round(float(p), 4) for p in probs]
            f["value"] = round(float(value), 3)
            f["sal"] = [round(float(s), 3) for s in sal]
            f["names"] = self.names
        self.buf.setdefault(i, []).append(f)

        # Live view: publish the WATCHED env's current frame every step so the
        # dashboard shows the game as it plays. Which env that is comes from a
        # tiny file the dashboard writes, so any of the N games can be watched
        # without shipping N full screens per step — at 48 envs that would be
        # ~200KB of writes per step onto /mnt/c, which is slow enough to throttle
        # training itself.
        if i == self.watched():
            self._write(LIVE, {**f, "env": i, "step": len(self.buf[i]),
                               "variant": self.variant, "names": self.names})

    def finish(self, i, env, info, started):
        frames = self.buf.pop(i, None)
        if not self.enabled or not frames:
            return
        solved = env.outcome.startswith("reached")
        if i != self.sample_env and not solved:
            return

        gid = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-rl{self.variant}{i}"
        (REPLAYS / f"{gid}.jsonl").write_text(
            "\n".join(json.dumps({**f, "game": gid}) for f in frames) + "\n",
            encoding="utf-8")

        with open(GAMES, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "ts": datetime.now().isoformat(timespec="seconds"),
                "agent": "rl-ppo", "turns": env.turns, "xl": env.xl,
                "depth": env.max_depth, "death": env.outcome,
                "score": env.turns, "actions": env.steps, "game": gid,
                "ttyrec": "", "source": "rl",
            }) + "\n")

        # Drop the oldest replays. Unbounded, this fills the disk overnight.
        files = sorted(REPLAYS.glob("*.jsonl"))
        for p in files[:-self.keep]:
            try:
                p.unlink()
            except OSError:
                pass

    def drop(self, i):
        self.buf.pop(i, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=8)
    ap.add_argument("--rollout", type=int, default=48, help="steps per env per update")
    ap.add_argument("--updates", type=int, default=400)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    # 0.02 was not enough to stop a full policy collapse: variant a reached
    # entropy 0.009 out of a possible 2.30 and could no longer explore its way
    # out of a local optimum. More entropy pressure is cheap insurance.
    ap.add_argument("--ent-coef", type=float, default=0.04)
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--target-depth", type=int, default=5)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--reset-heads", action="store_true",
                    help="keep a warm-start encoder but reset actor/critic "
                         "for exploratory PPO transfer")
    ap.add_argument("--no-record", action="store_true",
                    help="skip saving watchable replays")
    ap.add_argument("--keep-replays", type=int, default=40)
    ap.add_argument("--variant", choices=sorted(VARIANTS), default="a",
                    help="equipment handling: a=env picks item, "
                         "b=agent picks item, c=env does everything")
    args = ap.parse_args()
    set_paths(args.variant)
    action_names = [n for n, _ in VARIANTS[args.variant]]

    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    random.seed(0)

    # The update was measured at 14.7s on 3 CPU threads against ~12s of actual
    # play — i.e. over half the wall clock was matmuls, not the game. Rollout
    # buffers stay on the CPU (they are large and only read once per epoch);
    # only the minibatch crosses to the GPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = Policy(len(action_names))
    if args.resume:
        # Prefer this variant's own checkpoint; otherwise warm-start from the
        # shared 6-action policy. The screen encoder transfers — that is most
        # of what was learned — but the actor head has a different width, so it
        # starts fresh. Expect a dip before it beats the old solve rate.
        src = CKPT if CKPT.exists() else DATA / "rl_policy.pt"
        if src.exists():
            sd = torch.load(src)
            report = load_policy_state(policy, sd)
            print(f"resumed from {src.name}: loaded {len(report['loaded'])} "
                  f"tensors, expanded {report['expanded']}, "
                  f"reinitialised {report['skipped']}")
            if args.reset_heads:
                # A tiny supervised curriculum can (correctly) become nearly
                # certain about its six fixtures, yet be disastrously certain
                # on an unseen dungeon. Preserve its glyph/menu representation
                # but let PPO begin with a broad action distribution and a
                # fresh value estimate in the real game.
                nn.init.orthogonal_(policy.actor.weight, gain=0.01)
                nn.init.zeros_(policy.actor.bias)
                nn.init.orthogonal_(policy.critic.weight, gain=1.0)
                nn.init.zeros_(policy.critic.bias)
                print("reset actor/critic heads after warm-start", flush=True)
    policy.to(device)
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"variant {args.variant} · params {n_params/1e6:.3f}M · device {device} · "
          f"{len(action_names)} actions: {action_names}", flush=True)

    opt = torch.optim.Adam(policy.parameters(), lr=args.lr, eps=1e-5)

    envs = [DCSSEnv(env_id=i, target_depth=args.target_depth,
                    max_steps=args.max_steps, variant=args.variant)
            for i in range(args.envs)]
    pool = ThreadPoolExecutor(max_workers=args.envs)
    obs = list(pool.map(lambda e: e.reset(), envs))
    cols = [e.color_text() for e in envs]

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    rec = Recorder(enabled=not args.no_record, keep=args.keep_replays,
                   names=action_names, variant=args.variant)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    ep_returns, ep_depths, ep_outcomes = deque(maxlen=60), deque(maxlen=60), deque(maxlen=60)
    running = [0.0] * args.envs
    t_start = time.time()
    previous_nonsense_total = 0
    previous_nonsense_actions = Counter()
    total_steps = 0

    for update in range(1, args.updates + 1):
        buf_obs, buf_act, buf_logp, buf_val, buf_rew, buf_done, buf_mask = \
            [], [], [], [], [], [], []
        acts_taken = Counter()

        for _ in range(args.rollout):
            x = torch.stack([encode(o) for o in obs])
            # Read the mask before acting: it describes the same observation
            # whose log probability is stored in this PPO transition.
            action_mask = torch.tensor([e.action_mask() for e in envs],
                                       dtype=torch.bool)
            with torch.no_grad():
                xd = x.to(device)
                a, logp, _, v = policy.act(xd, action_mask.to(device))
                a, logp, v = a.cpu(), logp.cpu(), v.cpu()
                # What the net computed for this exact decision, saved with the
                # frame so the replay viewer needs no inference of its own.
                if rec.enabled:
                    ex_p, ex_v, ex_s = policy.explain(xd, action_mask.to(device))
                    ex_p, ex_v, ex_s = ex_p.cpu(), ex_v.cpu(), ex_s.cpu()
                else:
                    ex_p = ex_v = ex_s = None

            def do(i):
                return envs[i].step(int(a[i]))
            results = list(pool.map(do, range(args.envs)))

            rews, dones, next_obs, next_cols = [], [], [], []
            for i, (o2, r, d, info) in enumerate(results):
                acts_taken[info["action"]] += 1
                running[i] += r
                rews.append(r)
                dones.append(float(d))
                rec.step(i, obs[i], cols[i], info["action"], time.time() - t_start,
                         None if ex_p is None else ex_p[i],
                         None if ex_v is None else ex_v[i],
                         None if ex_s is None else ex_s[i])
                if d:
                    ep_returns.append(running[i])
                    ep_depths.append(envs[i].max_depth)
                    ep_outcomes.append(envs[i].outcome)
                    running[i] = 0.0
                    # Record BEFORE reset: reset() clears the episode's stats.
                    rec.finish(i, envs[i], info, t_start)
                    o2 = envs[i].reset()      # auto-reset; value bootstrap handles it
                next_obs.append(o2)
                # Read colours AFTER any reset, so they describe the same
                # screen as o2 rather than the episode that just ended.
                next_cols.append(envs[i].color_text())

            buf_obs.append(x)
            buf_act.append(a)
            buf_logp.append(logp)
            buf_val.append(v)
            buf_rew.append(torch.tensor(rews, dtype=torch.float32))
            buf_done.append(torch.tensor(dones, dtype=torch.float32))
            buf_mask.append(action_mask)
            obs = next_obs
            cols = next_cols
            total_steps += args.envs
            rec.publish_envs(envs)

        # --- GAE ---
        with torch.no_grad():
            _, last_v = policy(torch.stack([encode(o) for o in obs]).to(device))
            last_v = last_v.cpu()
        vals = torch.stack(buf_val)                       # T, N
        rews = torch.stack(buf_rew)
        dones = torch.stack(buf_done)
        adv = torch.zeros_like(rews)
        gae = torch.zeros(args.envs)
        for t in reversed(range(args.rollout)):
            nextv = last_v if t == args.rollout - 1 else vals[t + 1]
            # `done` masks BOTH the bootstrap and the trace, so a finished
            # episode never leaks value into the one that replaces it.
            delta = rews[t] + args.gamma * nextv * (1 - dones[t]) - vals[t]
            gae = delta + args.gamma * args.lam * (1 - dones[t]) * gae
            adv[t] = gae
        ret = adv + vals

        b_obs = torch.cat(buf_obs)
        b_act = torch.cat(buf_act)
        b_logp = torch.cat(buf_logp)
        b_mask = torch.cat(buf_mask)
        b_adv = adv.reshape(-1)
        b_ret = ret.reshape(-1)
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

        n = b_obs.shape[0]
        idx = torch.arange(n)
        pl = vl = el = 0.0
        for _ in range(args.epochs):
            idx = idx[torch.randperm(n)]
            for s in range(0, n, args.minibatch):
                mb = idx[s:s + args.minibatch]
                mb_obs = b_obs[mb].to(device)
                mb_act, mb_logp = b_act[mb].to(device), b_logp[mb].to(device)
                mb_adv, mb_ret = b_adv[mb].to(device), b_ret[mb].to(device)
                mb_mask = b_mask[mb].to(device)
                logits, v = policy(mb_obs)
                dist = torch.distributions.Categorical(
                    logits=apply_action_mask(logits, mb_mask))
                logp = dist.log_prob(mb_act)
                ratio = (logp - mb_logp).exp()
                a1 = ratio * mb_adv
                a2 = torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * mb_adv
                p_loss = -torch.min(a1, a2).mean()
                v_loss = F.mse_loss(v, mb_ret)
                ent = dist.entropy().mean()
                loss = p_loss + args.vf_coef * v_loss - args.ent_coef * ent
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                opt.step()
                pl, vl, el = p_loss.item(), v_loss.item(), ent.item()

        # --- report ---
        n_ep = len(ep_depths)
        mean_ret = sum(ep_returns) / max(1, len(ep_returns))
        mean_depth = sum(ep_depths) / max(1, n_ep)
        best = max(ep_depths) if n_ep else 0
        solved = sum(1 for d in ep_depths if d >= args.target_depth) / max(1, n_ep)
        el_s = time.time() - t_start
        top = ", ".join(f"{k}:{v}" for k, v in acts_taken.most_common(3))
        nonsense_total = sum(e.nonsense_total for e in envs)
        nonsense_actions_total = Counter()
        for e in envs:
            nonsense_actions_total.update(e.nonsense_actions_total)
        nonsense_this = nonsense_total - previous_nonsense_total
        nonsense_actions_this = nonsense_actions_total - previous_nonsense_actions
        previous_nonsense_total = nonsense_total
        previous_nonsense_actions = nonsense_actions_total.copy()
        nonsense_top = ", ".join(
            f"{k}:{v}" for k, v in nonsense_actions_this.most_common(3)) or "none"
        print(f"u{update:04d} steps={total_steps} {total_steps/el_s:.1f}/s | "
              f"ret={mean_ret:+.2f} depth={mean_depth:.2f} best=D:{best} "
              f"D{args.target_depth}={solved*100:.0f}% | "
              f"ent={el:.3f} vloss={vl:.3f} | {top} | "
              f"nonsense={nonsense_this}/{args.envs * args.rollout} "
              f"({nonsense_top})", flush=True)

        with open(LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                # Stamp every row with the run that produced it. `update`
                # restarts at 1 on every resume, so without this the log silently
                # concatenates runs and any chart drawn from it snaps back to
                # x=1 mid-line.
                "run": run_id, "variant": args.variant,
                # equips now counts AC IMPROVEMENTS, not wear/wield messages —
                # the old message detector read 0 for an entire 11-hour run.
                # ac_gained is the same thing in points; mean_ac is the level
                # the live envs are actually walking around at, which is the
                # honest answer to "is the gear reaching the character".
                "equips": sum(e.equips for e in envs),
                "ac_gained": sum(e.ac_gained for e in envs),
                # Average over envs whose status panel was READABLE, not all of
                # them. `e.ac or 0` counted a covered panel as AC 0 and dragged
                # the mean below the starting AC 2, which read as "armour is
                # not reaching the character" while ac_gained was 15 — the same
                # unknown-treated-as-zero mistake the env itself is careful to
                # avoid when parsing status.
                "mean_ac": _mean([e.ac for e in envs]),
                # Weapon power. mean_wpn below 7 means the wield action is
                # handing away the starting hand axe again — that was the old
                # behaviour 23.5% of the time and it is the number that proves
                # the upgrade filter is working.
                "wpn_gained": sum(e.wpn_gained for e in envs),
                "mean_wpn": _mean([e.wpn for e in envs]),
                "equip_refused": sum(e.equip_refused for e in envs),
                # Picks that left the character measurably worse. Only variant b
                # can score here (a and c let the env choose), and it falling
                # over time is the evidence that the agent is learning to READ
                # the menu rather than stab at a slot.
                "bad_choices": sum(e.bad_choices for e in envs),
                # Menus opened and walked away from. Variant b
                # collapsed into doing only this — it cost nothing
                # and the stall detector could not see it.
                "menu_abandoned": sum(e.menu_abandoned for e in envs),
                "hits": sum(e.hits for e in envs),
                "kills": sum(e.kills for e in envs),
                # The number to watch: if nonsense does not fall, the penalty
                # is not teaching anything and should be cut rather than raised.
                "nonsense": sum(e.nonsense for e in envs),
                "nonsense_rollout": nonsense_this,
                "nonsense_rate": round(nonsense_this / (args.envs * args.rollout), 4),
                "nonsense_actions": dict(nonsense_actions_this),
                "ascents": sum(e.ascents for e in envs),
                "travel_refused": sum(e.travel_refused for e in envs),
                "berserks": sum(e.berserks for e in envs),
                "berserk_wasted": sum(e.berserk_wasted for e in envs),
                "update": update, "steps": total_steps, "elapsed_s": round(el_s),
                "mean_return": round(mean_ret, 3), "mean_depth": round(mean_depth, 3),
                "best_depth": best, "solve_rate": round(solved, 3),
                "entropy": round(el, 4), "v_loss": round(vl, 4),
                "p_loss": round(pl, 4), "episodes": n_ep,
                "actions": dict(acts_taken),
                "outcomes": dict(Counter(ep_outcomes)),
            }) + "\n")

        if update % 5 == 0:
            torch.save(policy.state_dict(), CKPT)

    torch.save(policy.state_dict(), CKPT)
    for e in envs:
        e.close()


if __name__ == "__main__":
    main()
