# Findings

Append to this as the project goes. The dashboard renders it. Newest first.

Format: `## YYYY-MM-DD — short title`, then a few sentences. Say what you
actually observed, and separate observation from guess. A finding that says
"probably" is more useful than one that pretends certainty.

---

## 2026-08-13 — The equipment reward never fired once, in 2,449 updates

`equips` read **0 in every single update of all three variants** for the whole
11.6-hour run — 486 updates for a, 764 for b, 1199 for c. Not "low". Zero.

The cause: the reward was paid on the message `You are now wearing ...`. Across
28,938 replay frames that string appears **0 times**, while `continue putting
on` appears 57 times. Wearing body armour is a multi-turn action, and the
completion line has already scrolled out of the message area by the time the
post-action screen is sampled. So the agent *was* putting armour on ~57 times
and was paid for it exactly never.

What this cost: **variant b's entire reason to exist.** b differs from a only
in having three wear/wield slots instead of one, and it was spending ~12% of
its actions on them (50–70 of ~512 per update, consistently top-3 in the mix)
for a reward branch that could not fire. The a/b/c comparison measured nothing.

The generalisable lesson, and the reason this one stings: an earlier note in
`dcss_env.py` had *already seen* `equips == 0`, and explained it as "the
character starts fully equipped and nothing new reaches the pack" — then
changed autopickup on the strength of that reasoning. The number was 0 because
its detector was broken. **A zero counter is evidence about the counter before
it is evidence about the world.** Any reward branch should be proven to fire at
all before its size is argued about.

Fixed by scoring the **result** rather than the attempt: AC is a real number in
the status panel (`AC:  2`), so equipment is now paid per point of AC gained.
Validated on 37,726 real frames — readable in 95.0%, never ambiguous (zero
frames matched twice), and values 3/5/6/8/9/10 appear in 2,339 of them. The
character *had* been getting into better armour ~6.5% of the time and was paid
nothing for any of it.

Credit is against a **high-water mark**, not a raw delta, because a raw delta is
farmable: armour comes off as easily as on, and variant b has three wear slots
to cycle. Same rule `max_depth` already uses.

## 2026-08-13 — `wield` was a downgrade machine: 23.5% of frames held a worse weapon

Classifying the wielded weapon across 54k frames (parsed from the status panel,
anchored to the AC column so map glyphs can't spoof it):

| holding | frames | |
|---|---|---|
| `+0 hand axe` — the starting weapon | 34,153 | 75.7% |
| non-axe melee (club, dagger, whip, mace, rapier) | ~9,900 | 22.6% |
| ranged (sling, shortbow) — useless in melee | 663 | 1.5% |
| a **better** axe (war axe, always `+0`) | 128 | 0.3% |

A message area caught in the act:

```
You unwield your +0 hand axe.
c - a +0 club (weapon)
```

A Minotaur Berserker — whose trained skill is Axes — trading a hand axe
(base 7) for a club (base 5, untrained Maces & Flails), on D:1 at XL 1.

This is **structural, not a tuning problem**. `_equip` excluded items marked
`(weapon)` and then picked the *nth offered item in inventory-letter order*.
The wielded axe is therefore never a candidate, and letter order is arbitrary,
so `wield` could only ever swap AWAY from the starting weapon, to something
chosen at random. Its expected value was strongly negative. No reward bolted on
top could have fixed it: with variant a's single `wield` action there was no way
for the policy to express *which* item it wanted.

Fixed in three parts, in this order:

1. **Make the action semantic.** Offered items are filtered to strict upgrades
   and sorted by power, so `wield1` means "the best weapon I carry that beats
   what I hold" in every game. This is also the first time variant b's three
   wield slots mean anything stable.
2. **Score axes on their own ladder, not by base damage.** The trap: a great
   sword rolls 13 to a war axe's 11, so scoring by raw damage would teach a
   Berserker with no Long Blades skill to throw the axe away. Every non-axe
   scores *below* the starting hand axe, so swapping off it is never an upgrade.
3. **Then** pay for it, high-water mark, same as AC.

Armour is deliberately not filtered this way: AC is measured, so a bad `wear`
simply goes unpaid. Weapons have no such ground truth, which is why they need
the ordering instead.

On whether better axes are reachable at all — they are, but barely: a war axe
was wielded in 128 frames, so they spawn and can be picked up by D:3. But no
broad axe, battleaxe, or *enchanted* axe was wielded in 54k frames, and branded
weapons (`+4 dagger (venom)`, `+2 dagger (drain)`, `+1 dagger (speed)`) were
reaching the pack and being passed over. The upgrades were there; the agent had
no way to see them.

## 2026-08-13 — `games.jsonl` overstates the solve rate 8x, by construction

Read naively, the file says **56.2% of RL episodes reached D:5**. The true
figure is **7.2%**. The trainer's own `D5=` line was right all along (5–17%);
the game log is the thing that lies.

`train_rl.py` logs every solve from all 16 envs, but non-solves only from env 0:

```python
if i != self.sample_env and not solved:
    return
```

So the file is enriched in wins by roughly the env count. Any analysis over
`games.jsonl` must filter to `rl{variant}0` or it is reading a highlight reel.
Full history, env 0 only:

| outcome | n | |
|---|---|---|
| step limit | 171 | 33.1% |
| stalled | 161 | 31.2% |
| died | 145 | 28.1% |
| **reached D:5** | **37** | **7.2%** |

This is a sampling-bias bug in the *instrumentation*, not the agent, and it is
the second one found today with the same shape as the `equips` bug: a number
that was never measuring what its name said.

## 2026-08-13 — The two failure modes split cleanly by variant

Over the last 50 env-0 episodes, a/b and c fail in different ways, which
suggests two separate problems rather than one plateau:

- **a and b die.** 20 of 33 episodes, clustered at D:2–4 at **XL 1–3** — badly
  underlevelled (a human is XL 7–9 by D:4). They descend faster than they
  level, and `rest` barely appears in the action mix. The v4 death penalty
  (-0.5 → -3.0) did not fix this.
- **c burns the clock.** 13 of 17 episodes end in step limit or stall. The
  giveaway is turns-per-action: `1000 actions → 11 game turns`, `→38`, `→85`,
  `→144`. **Up to 99% of its actions consume no game time at all** — menus,
  blocked explore, refused travel. That is a wedge, not slow play. c also has
  the lowest entropy (1.26 vs ~1.9/2.1) and sat at `depth=2.05, D5=5%` for
  100+ updates: converged onto a degenerate explore/autofight loop.

Related: `nonsense` did not fall over the run (a: 1394→1727, b: 1405→1815 per
update), and `travel_refused` held near 300/update throughout. `dcss_env.py`'s
own comment names exactly this as the trigger to cut the penalty rather than
raise it — worth acting on if the rate stays flat under the new rewards.

---

## 2026-08-11 — D:5 reached by a policy trained only on its own games

Reward v3, PPO, update 104 (53,248 steps, 63 minutes from random init):

```
best=D:5   D5 solve=2%   mean_depth=1.93   entropy=1.382   return=-0.43
```

**D:5 is the target the whole project was aimed at**, and it was reached by a
0.116M-parameter network that has never seen a teacher trace — it reads the
80x24 terminal screen and picks one of six macro-actions, trained purely on
reward from its own play. Entropy 1.382 is 23% below ln(6)=1.792, so this is
not a random walk that got lucky.

Honesty about what it is *not*: a **2% solve rate is one episode in a
60-episode window**. This is proof the objective is reachable under this
reward, not a reliable D:5 clearer. The claim to defend is "the policy learns
and has solved it", not "the policy solves it".

The healthy-learning signature, contrasted with run 2's reward hacking — here
depth and return rose *together*:

| update | mean depth | return | entropy |
|---|---|---|---|
| 13 | 1.75 | −2.28 | 1.681 |
| 61 | 1.88 | −0.85 | 1.612 |
| 104 | **1.93** | **−0.43** | **1.382** |

## 2026-08-11 — 28 cores, load average 0.44: the run was using 2% of the machine

For 100 updates this trained at 14 steps/s on 8 parallel games while `nproc`
reported **28** and load average sat at **0.44**. Raising to 24 envs took
throughput to **51.6 steps/s — 3.7x — for free**, with no algorithm change.

Because the reward was unchanged, `--resume` carried the learned weights
across the restart, so the 100 updates were not thrown away (confirmed by
entropy loading at 1.549 rather than 1.79).

Worth remembering: the environment here is pty I/O, not compute, so the
parallel-env count should be sized against *idle cores*, not against the model.
Nothing in the training logs would ever have revealed this — the only symptom
was a number that looked fine in isolation. **Check machine utilisation early;
a 3.7x speedup was sitting unused for an hour.**

## 2026-08-11 — Two reward failures, one root cause: dense shaping beat the sparse goal

Run 2 fixed run 1's passivity and then failed the opposite way. The tell was
**return improving while depth fell** — the agent was optimising successfully,
just not the objective:

| update | mean depth | mean return | explore | autofight | escape |
|---|---|---|---|---|---|
| 11 | 2.10 | −9.79 | 10% | 13% | 4% |
| 26 | 1.74 | −9.07 | 4% | 36% | 1% |
| 37 | **1.70** | **−8.56** | 10% | 45% | 0% |

The cause is arithmetic, and it needed no extra data to diagnose:

* no-op penalty budget per episode: `0.05 × 300 steps` = **15 points**
* the entire objective, D:1 → D:5: `4 levels × 2.5` = **10 points**

The anti-idling term was worth more than solving the game, so the optimal
policy is "always press a key that does something" — which is precisely what
it learned (`escape` → 0%, `rest` → 5%, `autofight` → 45%: spam the
always-valid action).

Both failures are the same error with opposite signs: **a dense per-step
shaping term outweighed a sparse terminal objective.** v1 made passivity
optimal via HP shaping; v2 made frantic key-mashing optimal via the no-op
penalty.

**The invariant now written into `dcss_env.py`, and the first thing to check
if it fails again:** worst-case total shaping cost of an episode must be
comfortably smaller than the reward for solving it. v3 is `300 × (0.01+0.02)
= −9` against `4 × 5.0 + 15 = +35`. Measured separation under a **random**
policy: D:3 episodes scored +7.2/+7.4/+8.2, D:1 episodes −1.9/−2.3/−2.5.

Also worth keeping: **the action histogram diagnosed both failures in
seconds, and the return curve diagnosed neither.** In run 1 return fell, which
looks like ordinary early-training noise; in run 2 return *rose*, which looks
like success. Log the action distribution every update.

## 2026-08-11 — The first reward function taught the agent to do nothing

Run 1 of PPO learned a degenerate policy and the action mix shows it cleanly:

| | update 3 | update 38 |
|---|---|---|
| explore | 11% | **3%** |
| autofight | 17% | **2%** |
| escape | 12% | **32%** |
| mean depth | 1.00 | 1.55 (peak 1.75 at u7) |
| mean return | −1.38 | −1.93 |

46 consecutive episodes ended at the step limit: it survived 300 steps by
standing still. Two shaping terms caused it, and both were mistakes of kind,
not of tuning:

1. **`+k · Δhp_frac` punishes every fight.** Winning a fight still costs HP, so
   `autofight` is locally negative *always*. Passivity strictly dominates.
2. **A large death penalty (−2) makes cowardice optimal.** Dying already
   forfeits every future depth reward via termination. Charging for it a
   second time drowns out the objective.

Compounding both: `escape` advances no game turn, so it was completely free —
no HP loss, no death risk, −0.01 and done.

Reward v2 drops HP shaping entirely, cuts death to −0.5, charges −0.05 for any
action that fails to advance a game turn, and raises depth to +2.5. Measured
separation under a **random** policy afterwards: episodes reaching D:2 scored
−1.25/−1.40, episodes stuck on D:1 scored −4.55/−4.80. A 3.4-point gap for one
level, with no way to score by idling.

Generalisable: **potential-based-looking shaping on a survival quantity (HP)
inverts the objective in any game where progress requires risk.** The action
histogram found it in seconds; the return curve alone would not have — return
was drifting down, which looks like ordinary early-training noise.

## 2026-08-11 — RL is affordable here, and the reason is the action space

Earlier in this project I wrote off RL as infeasible on this hardware. That
judgement was wrong, and it was wrong for a specific reason worth recording:
**I was costing RL over raw keystrokes.** Over DCSS's own macro commands the
problem is a different size entirely.

`o` (auto-explore) is one decision worth hundreds of game turns. `Tab` is an
entire fight. `X > Enter` is travel across a level. With a six-action macro
space, an episode that reaches D:5 is **~200 decisions, not ~10,000
keystrokes** — two orders of magnitude off the NetHack-Challenge-scale problem
I was implicitly comparing against.

The measured precondition for bootstrapping: a **uniformly random** policy
reached **D:2 in 3 of 6 episodes** and gained XL. Reward is reachable by
chance, which is exactly what PPO needs and what raw-keystroke RL would not
have given.

Throughput: **10–16 macro-steps/s** across 4–8 parallel pty games. Torch runs
on **CPU** deliberately — the bottleneck is the game process, not the matmul,
and it leaves the 8GB card free.

## 2026-08-11 — Two silent env bugs, both of which faked a healthy game

Building `dcss_env.py` surfaced two failures that produced *no error at all*:

1. **A missing `saves/` directory wedges crawl.** With `-dir` pointing at a
   fresh directory containing only `morgue/` and `rcs/`, crawl starts, writes
   ~1900 bytes of terminal-init sequence, and then blocks forever — alive, no
   error, no output, never exits. Creating `saves/` fixes it.
2. **`drain()` could not tell "nothing yet" from "finished drawing".** The
   read loop returned once the pty was quiet for `quiet` seconds, but the
   timer started at call time, so a slow-starting game returned a **blank
   screen as a legitimate observation**. Fixed with a `got_any` guard.

Both presented identically — `turns=0` forever with every process healthy —
and the same latent flaw exists in `pty_agent.read_until_quiet`, which has
only ever been saved by warm-cache timing.

Diagnostic that settled it: running the known-good `pty_agent.Crawl` and the
new `_Crawl` **side by side in one process**, then diffing the captured argv
and env dicts programmatically. That reduced a vague "mine doesn't work" to a
single differing token (`-dir`) in about a minute. Worth reaching for whenever
a reimplementation fails and the original works.

## 2026-08-11 — First real learning: model reproduces the teacher at 96.5%

Trained on 2,640 teacher decisions (entropy 0.424).

| metric | majority baseline | model |
|---|---|---|
| val_top1 | 0.6808 | **0.9654** |
| val_action_loss | 0.3358 | 0.4665 |

The model genuinely learned the policy — 96.5% of the teacher's keys
reproduced from the screen alone, against a 68% floor. That is the first
learnable signal this project has produced.

It also overfits hard: training loss reached **0.0000** (~300 epochs over 2,349
samples), so the loss metric is worse than the trivial baseline even while
accuracy is far better. Same accuracy/loss disagreement as the synthetic run,
same cause: right more often, but wildly overconfident when wrong. Needs more
data and regularisation, not a better architecture.

## 2026-08-11 — Getting the teacher to descend: three dead ends and a fix

`>` alone does **not** auto-travel in this build — it answers "You can't go
down here!" unless you are already standing on stairs. The teacher pressed it
208 times in one run. `G` is not a valid key at all ("Unknown command"). What
works is the level map: **X** opens it, **>** jumps the cursor to the next down
staircase, **Enter** travels there, then **>** descends. With that, the teacher
reaches D:2–D:4 routinely (best so far: D:4 at XL 5).

Also fixed: autofight refuses below the `autofight_stop` HP fraction ("You are
too injured to fight recklessly!") and silently eats the keypress, so a
fight-whenever-a-monster-is-visible policy loops on Tab forever — 141 times in
one run. Set `autofight_stop=0` and added a guard so no key can be repeated
into a wall.

## 2026-08-11 — Label the policy, and label the flailing

Two data-hygiene bugs that would have quietly ruined the dataset:

1. **pty traces did not record which policy chose the key.** Once merged,
   teacher and random traces are indistinguishable — and random labels are not
   a function of the screen, so blending them in makes the whole set
   untrainable. Now every trace carries `policy`.
2. **The unwedge cycle looked like decisions.** When the policy can't make
   progress it rotates through esc/5/o/enter/>, which produces five keys at
   ~18% each — a near-uniform distribution that swamps real signal. With those
   flagged and excluded, entropy fell **0.866 → 0.424** and the prompt-key
   share fell **31% → 2.9%**.

Rows from before those fields existed are excluded rather than assumed clean:
absent means unknown, and unknown is not trustworthy.

## 2026-08-10 — Collector workers could hang forever at a prompt (fixed)

Two of twelve workers ran **25+ minutes without completing a single game**,
against a cap that should finish one in ~20 seconds. Both sat in `poll`, their
crawl children alive but silent.

Cause was mine, in `pty_agent.py`: the branches that answer forced UI prompts
`continue` **without incrementing `actions`**, so the `actions < max_actions`
cap could never terminate the loop. If crawl sits at a prompt the keypress
doesn't clear, the worker spins indefinitely — alive, busy, and producing
nothing.

Fixed with a global iteration guard (`max_actions * 6`) and a distinct
`"stuck at prompt"` outcome so it shows up in the data instead of hiding.

Worth generalising: **a bounded counter is only a bound if every path through
the loop increments it.** The failure mode was invisible from outside — the
process looked healthy and busy the whole time. What exposed it was comparing
elapsed time against expected duration, not any error.

## 2026-08-10 — The wrapper was never needed; ~200 lines replaced ~10,000

`dcss-ai-wrapper` reconstructs full game state in order to choose moves, and
its parser does not understand DCSS 0.35. But **driving** a game needs none of
that — log in, start, send keys. `webtiles_agent.py` does exactly that in ~200
lines and works against current trunk.

Better still, it gives the split we actually wanted: the webtiles protocol
ships *state* (`map` cells with glyphs, `player` stats) and the **browser**
does the drawing. So one game serves both — a human watches full tiles at
`localhost:8090` while the agent reads the same messages as text. No graphics
pipeline on the agent side, no duplicated rendering.

Two traps found while building it:
- Frames are **raw-deflate with the trailing `00 00 FF FF` stripped**, and
  decompression needs ONE persistent zlib context for the whole connection. A
  fresh context per message fails; the symptom is 100% undecodable frames.
  Setting `use_gzip = False` in the server config does *not* turn it off.
- Shell scripts written from Windows carry CRLF, which turns `cd /some/dir`
  into `cd /some/dir\r`. It fails **silently** and the script continues from
  the wrong directory. This is one of several reasons the .sh files are gone.

## 2026-08-10 — Parallel collectors must not share a crawl `-name`

First fleet run: 6 workers, all with the default `--name bot`, so all six
shared one save file. Result: 36 of 46 games returned **turn 0** — they stomped
each other. Nothing errored; the games simply came back empty, and every
dashboard panel still looked healthy.

Two fixes, both in `ops.py fleet`: a distinct `-name` per worker, and a
distinct `--tag` so concurrent multi-KB JSON appends can't tear each other's
lines in a shared file. With those, workers return turns of 123–3,580.

Related quirk worth knowing: a worker reuses its save between games, so its
"games" are really consecutive sessions of one character and the turn counter
climbs across them. That yields *more varied* states (deeper levels, more
monsters) than always restarting at D:1, so it was left as is — but the rows
are not independent games and shouldn't be treated as such in any analysis.

## 2026-08-10 — "Idle agent" was four bugs, three fixed, one structural

The agent connected, started a game and then sat at turn 0 forever. Causes, in
the order they were peeled back:

1. **No agent attached.** `self.agent` was never set and the send path is
   guarded by `if self.agent:`. Upstream's demo has `load_ai_agent()`
   commented out. Fixed in `run_agent.py` via `set_ai_class` + `load_ai_class`.
2. **Wrong message format at character creation.** The bundled agent sends
   `{'msg':'key','keycode':…}`; webtiles wants `{'msg':'input','text':…}`.
   Fixed with a `WebSimpleRandomAgent` subclass overriding
   `get_game_mode_setup_actions`.
3. **Character creation menus unparseable on trunk.** Bypassed entirely with a
   `bot-web-trunk` game definition passing `-species Minotaur -background
   Berserker -extra-opt-first weapon="hand axe"`. This works — `hp_max` goes
   0 → 19 and zero menus are detected.
4. **Protocol drift, not yet solved.** `check_received_map_data` waits for a
   `{"msg": "map"}` message, and DCSS 0.35 never sends one. Trunk sent only:
   html, options, layout, set_game_links, ping, version, ui_state, player,
   update_spectators, login_success, lobby_*, game_started, game_client, chat.
   The action-send branch is gated on `_RECEIVED_MAP_DATA`, so the agent is
   never once asked for a move.

Root cause of 3 and 4 is the same: `dcss-ai-wrapper`'s own docstring targets
**"crawl 23.1"** and we built **0.35-a0 trunk**. Bypassing creation moved the
wall from the menu layer to the protocol layer; it did not remove it. The real
fix is almost certainly building crawl at a tag from the wrapper's era.

Useful diagnostic that settled it: counting message types actually received
(`grep -oE '"msg": *"[a-z_]+"' | sort | uniq -c`) against what the code waits
for. Cheaper than reading either codebase.

## 2026-08-10 — Dashboard hung in the browser but not in scripts

The control panel would load fine from `Invoke-WebRequest` and hang forever in
a real browser. Cause: `http.server.HTTPServer` is **single-threaded**, and
browsers open speculative connections they leave idle. The handler blocks
reading a request line that never comes, and the accept loop stops entirely.
The 10s auto-refresh then queued behind it, so it never recovered.

Fixed with `ThreadingHTTPServer`, a 15s read timeout, and a 3s cache over the
probe results (probing costs ~2.4s; several simultaneous requests each paid it
in full). Cold 4.2s, warm 0.01s.

Worth remembering: **a scripted HTTP client cannot reproduce this class of bug.**
One connection, opened and closed, never triggers it. Test web UIs with a
browser.

## 2026-08-10 — Environment is fully built end to end

DCSS compiles and runs, webtiles serves on port 8090, `dcss-ai-wrapper`
installs and imports. From cold Windows to a playable local server took one
reboot and three bug fixes (below). No agent has played a real game yet.

## 2026-08-10 — Crawl's build docs are stale on Ubuntu 26.04

Four package names in the official build instructions no longer exist:
`libncursesw5-dev` → `libncurses-dev`, `libz-dev` → `zlib1g-dev`,
`libfreetype6-dev` → `libfreetype-dev`, `ttf-dejavu-core` → `fonts-dejavu-core`.
Failure mode is a clean apt error, so it's obvious once you look.

## 2026-08-10 — Upstream parallel-build race in the webtiles target

`make -j28 WEBTILES=y` fails with *"No rule to make target
webserver/game_data/static/status-icon-sizes.js"*. The rule that nominally
builds that `.js` is a **no-op** — the file is really a side effect of the
`status-icon-sizes.h` rule running a Python generator. Under parallel make the
webserver target can demand it before the generator has run. Workaround in
`build_fix.sh`: run the generator, copy the file into place, resume. Would
probably not reproduce at `-j1`; not verified.

## 2026-08-10 — Python 3.14 DOES break this stack (corrects an earlier note)

An earlier version of this file said the Python 3.14 risk "didn't
materialize." **That was wrong**, and only looked right because the check was
too narrow — `dcss-ai-wrapper` *installs* fine on 3.14, so the pip step passed.
The breakage is elsewhere:

1. `main_webserver.py` calls `asyncio.get_event_loop()` expecting it to create
   a loop. Removed in 3.12+; now raises `RuntimeError: There is no current
   event loop`. Fixed in our own `run_agent.py`, which creates the loop before
   constructing the autobahn factory.
2. **Crawl's own webtiles server is not 3.14-compatible.** It throws
   `RuntimeError: loop <...> is not the running loop` from its asyncio code on
   every message, so the Crawl subprocess never starts. Fixed by rebuilding
   `~/webtiles-venv` on **Python 3.12** via uv.

Worth remembering how it presented: the only client-visible symptom was
`Error while starting the Crawl process!` — nothing about Python, asyncio, or
versions. The real traceback existed solely in the server's stdout, which
wasn't being captured. **Restarting the server with a log is what solved it.**
Generalisable lesson: when a subprocess "fails to start," go read the parent's
log before theorising.

## 2026-08-10 — Webtiles forks a child that also holds the port

`pkill -f webserver/server.py` on the parent leaves port 8090 bound by the
child. The next server then dies instantly with `OSError: [Errno 98] Address
already in use`, which is easy to miss because the *old* server is still
serving — so the site looks up while your new config isn't running.
`restart_stack.sh` now kills by whoever holds the socket (`ss -ltnp`) and waits
for the port to clear before rebinding.

## 2026-08-10 — Port 8080 collision (deliberate divergence)

llama-server (the local Qwen) owns 8080, which is also webtiles' default.
Webtiles moved to **8090** and the wrapper's `config.py` was patched to match.
If either is ever reinstalled from defaults, this is the first thing to check.

## 2026-08-10 — Baseline model overfits hard on the synthetic set

On 3,600 synthetic samples, the starter model reached `val_top1=0.7075`
(majority baseline 0.3825) while `val_action_loss` went to **2.1583** —
*worse* than the 0.8943 baseline. Training loss fell to ~0.2, i.e. ~214 epochs
of memorization.

The lesson generalizes past the synthetic data: **accuracy and loss disagree
here, and accuracy is the misleading one.** A model that's right more often but
wildly overconfident when wrong scores better on top-1 and worse on the metric.
Any dashboard or log reporting only accuracy would have called this a success.
Generate far more data before drawing conclusions.

## 2026-08-10 — Qwen releases VRAM when idle

Earlier assumption: the local Qwen (7.6GB) and a training run cannot share the
8GB card, full stop. Observed: while idle its VRAM is evicted (down to ~1.1GB),
and it reloads on the next request. So an *idle* Qwen doesn't block training —
only a busy one does. Mechanism is probably Windows WDDM evicting idle
allocations; inferred, not confirmed.
