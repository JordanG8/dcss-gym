# DCSS agent project

Train a small model to predict a competent DCSS player's next keystroke, using
Karpathy's autoresearch loop for the training side.

## The files

| file | what |
|---|---|
| `project.py` | dashboard + control panel (`--serve`) |
| `ops.py` | start/stop services, run collector fleets, merge shards |
| `webtiles_agent.py` | plays a **watchable** game through webtiles |
| `pty_agent.py` | plays fast headless games for **bulk data** |
| `autoresearch/prepare_dcss.py` | data + the metric (read-only for the agent) |
| `autoresearch/train_dcss.py` | the model the research agent edits |
| `autoresearch/program_dcss.md` | instructions for the research agent |
| `FINDINGS.md` | research log, newest first |

There are deliberately **no shell scripts**. Everything that used to live in a
pile of `.sh` files is in `ops.py`, so there's one place to look.

## Daily use

Double-click **DCSS Dashboard** on the Desktop, or:

```
python project.py --serve      # http://localhost:8099
```

From the panel you can collect games, start a watchable game, run a training
experiment, merge shards, read logs, and stop anything running.

From the command line:

```
python ops.py webtiles start
python ops.py fleet --workers 6 --games 20 --prefix w
python ops.py merge
python ops.py status
```

## Watching

- **Live:** run a watchable game, then open <http://localhost:8090> and click
  the player. Full webtiles UI, real tiles.
- **Past games:** the ◉ button on any tagged row replays the stored screens.

The agent never sees graphics — it reads glyphs and stats. The browser draws
the pictures. Same game, two views.

## Two things that will bite you

**Parallel collectors need distinct names.** Workers sharing a crawl `-name`
share a save file and return turn-0 games with no error. `ops.py fleet` handles
this; don't hand-roll it.

**Random data teaches nothing.** The dashboard's data-health panel measures
label entropy. At ~1.0 the labels are indistinguishable from random, which is
exactly what a random agent produces — no model can learn a mapping that
doesn't exist. Imitation learning needs a teacher worth imitating. The current
traces are useful for validating the pipeline, not for producing a good model.

## Next step

Write a rule-based teacher (approach visible monsters, auto-explore otherwise,
retreat at low HP) against the glyph grid `webtiles_agent.py` already builds,
then collect from that instead. Entropy should drop well below 1.0, and the
training loop finally has signal to find.
