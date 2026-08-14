"""A polished local companion for live WebTiles and recorded RL replays.

Live mode embeds the real WebTiles client, so it is exactly the same renderer a
human uses. Replay mode renders the recorded *player-visible* terminal frame
with its original console colours plus policy diagnostics; replay JSONL does
not contain WebTiles tile IDs, so it intentionally never invents hidden tiles.

    /root/pty-venv/bin/python spectator.py
    # http://127.0.0.1:8101
"""
import argparse
import json
import os
import signal
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from checkpointing import manifest_path, publish_manifest

HERE = Path(__file__).parent
DEFAULT_DATA = HERE / "data"
POLICY_AGENTS = 2
POLICY_USERS = tuple(f"midcai{i + 1}" for i in range(POLICY_AGENTS))
POLICY_LANES = ("candidate", "champion")


def is_tunnel_request(headers):
    """Cloudflare adds this only to requests arriving through its edge."""
    return bool(headers.get("CF-Connecting-IP"))

PAGE = r"""<!doctype html><meta charset="utf-8"><title>DCSS Spectator</title>
<style>
:root{color-scheme:dark;--ink:#d9ddd5;--muted:#92a098;--edge:#29362e;--pane:#101713;--accent:#9fd36b;--hot:#f4c66a}*{box-sizing:border-box}body{margin:0;background:#080d0a;color:var(--ink);font:14px system-ui,sans-serif}header{height:56px;display:flex;align-items:center;gap:14px;padding:0 22px;border-bottom:1px solid var(--edge);background:#101713}h1{margin:0;font-size:17px;letter-spacing:.4px}h1 b{color:var(--accent)}button,select,input{background:#18231d;color:var(--ink);border:1px solid #35473b;border-radius:6px;padding:7px 10px}button:hover{border-color:var(--accent);cursor:pointer}.wrap{display:grid;grid-template-columns:minmax(560px,1fr) 340px;gap:14px;padding:14px;max-width:1700px;margin:auto}.pane{border:1px solid var(--edge);border-radius:10px;background:var(--pane);overflow:hidden}.bar{padding:10px 12px;border-bottom:1px solid var(--edge);display:flex;gap:8px;align-items:center;flex-wrap:wrap}.bar .spacer{flex:1}.live{width:100%;height:calc(100vh - 160px);min-height:610px;border:0;background:#000}.replay{display:none}.term{margin:0;padding:14px;background:#020403;min-height:576px;overflow:auto;font:15px/1.12 ui-monospace,Consolas,monospace;white-space:pre;letter-spacing:0}.term span{white-space:pre}.w{color:#d7ddd4}.r{color:#ec6b6b}.g{color:#83d17e}.y{color:#d6b565}.b{color:#80a6ef}.m{color:#d987d9}.c{color:#71d7d5}.k{color:#424d44}.W{color:#fff}.R{color:#ff8b8b}.G{color:#aafa96}.Y{color:#ffe08b}.B{color:#a7c5ff}.M{color:#f2a5f1}.C{color:#9af4ef}.side{padding:12px}.muted{color:var(--muted)}.game{width:100%;text-align:left;margin:4px 0}.game small{display:block;color:var(--muted);margin-top:2px}.action{color:var(--hot);font-weight:700}.prob{display:grid;grid-template-columns:78px 1fr 38px;gap:7px;align-items:center;margin:5px 0;font-size:12px}.track{height:7px;background:#26352c;border-radius:9px;overflow:hidden}.fill{height:100%;background:#81b857}.picked .fill{background:var(--hot)}#diag{margin-top:12px;border-top:1px solid var(--edge);padding-top:10px}@media(max-width:1000px){.wrap{grid-template-columns:1fr}.live{height:65vh}.side{max-height:40vh;overflow:auto}}
.policy{padding:12px;margin:-2px -2px 14px;border:1px solid #35473b;border-radius:9px;background:#141f19}.policy h2{font-size:15px;margin:0 0 8px}.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:6px;margin:9px 0}.stat{background:#0b120e;border-radius:6px;padding:7px}.stat b{display:block;font-size:17px;color:var(--accent)}.badge{display:inline-block;border-radius:99px;padding:3px 8px;background:#29362e;color:var(--muted)}.badge.validating{color:var(--hot)}.badge.running{color:var(--accent)}.badge.failed{color:#ff8b8b}.buttons{display:flex;gap:7px}.buttons button{flex:1}.envs{padding:12px;margin:0 -2px 14px;border:1px solid #35473b;border-radius:9px;background:#0d1510}.envgrid{display:grid;grid-template-columns:repeat(4,1fr);gap:6px;margin:9px 0}.envcell{padding:7px 5px;text-align:left}.envcell.on{border-color:var(--hot);box-shadow:0 0 0 1px var(--hot) inset}.envcell b,.envcell small{display:block}.envcell small{color:var(--muted);margin-top:3px}.hpbar{height:4px;background:#26352c;border-radius:4px;overflow:hidden;margin-top:5px}.hpbar i{display:block;height:100%;background:var(--accent)}.hpbar i.low{background:#ec6b6b}
.probgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:3px 16px;margin-top:8px}#diag,#liveDiag{margin-top:0;padding:12px 14px;background:#0b120e;border-top:1px solid var(--edge)}.probgrid .prob{grid-template-columns:82px 1fr 38px;margin:4px 0}.policygrid{display:grid;grid-template-columns:repeat(4,1fr);gap:6px;margin:9px 0}.policycell{padding:7px 5px;text-align:left}.policycell.on{border-color:var(--hot);box-shadow:0 0 0 1px var(--hot) inset}.policycell b,.policycell small{display:block}.policycell small{color:var(--muted);margin-top:3px}
</style><header><h1><b>DCSS</b> Spectator</h1><button id="live">Live WebTiles</button><button id="openLive">Open WebTiles</button><button id="tileReplay">Tile Replays</button><button id="replay">Replay Lab</button><span class="muted">player-visible replays · local-only director diagnostics</span></header>
<main class="wrap"><section class="pane"><div class="bar"><b id="title">Live game — official WebTiles renderer</b><span class="spacer"></span><span id="playback"><button id="play">Play</button><label>speed <select id="speed"><option value="900">slow</option><option value="350" selected>normal</option><option value="100">fast</option></select></label></span></div><iframe id="webtiles" class="live"></iframe><div id="liveDiag"><span class="muted">Waiting for the selected neural agent's first decision…</span></div><div id="replayPane" class="replay"><pre id="term" class="term"></pre><input id="seek" type="range" min="0" value="0" style="width:100%"><div id="diag"></div></div></section><aside class="pane side"><div class="policy"><h2>Checkpoint WebTiles canaries</h2><span id="policyBadge" class="badge">offline</span><div id="policyGrid" class="policygrid"><span class="muted">loading…</span></div><div class="stats"><div class="stat"><b id="pd">—</b>depth</div><div class="stat"><b id="php">—</b>health</div><div class="stat"><b id="pt">—</b>turn</div></div><p id="policyText" class="muted">Candidate follows the latest atomic checkpoint; champion stays on the promoted best.</p><div class="buttons"><button id="startPolicy">Start canaries</button><button id="stopPolicy">Stop all</button><button id="policyReplay">Tiles</button></div></div><div class="envs"><b>Training environments</b><p class="muted">Headless actors feed the learner; select an environment when training is live.</p><div id="envGrid" class="envgrid"><span class="muted">no environments running</span></div><small id="envMeta" class="muted"></small></div><b>Recorded episodes</b><p class="muted">Live play is the authentic WebTiles page. Replays show exactly what the policy saw, including terminal colours and action probabilities.</p><div id="games"></div></aside></main>
<script>
const $=q=>document.querySelector(q);let frames=[],i=0,timer=null,mode='live',trainEnv=0,trainFrame=null,policySlot=0,policyAgents=[];
function esc(s){return String(s??'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))}
async function loadGames(){const gs=await (await fetch('/api/replays')).json();$('#games').innerHTML=gs.length?gs.map(g=>`<button class="game" data-id="${esc(g.id)}">${esc(g.id)}<small>${g.frames} frames · ${esc(g.variant||'legacy')}</small></button>`).join(''):'<p class="muted">No replay files found in this data folder.</p>';document.querySelectorAll('.game').forEach(b=>b.onclick=()=>openReplay(b.dataset.id));}
function setMode(next){mode=next;$('#webtiles').style.display=mode==='live'?'block':'none';$('#liveDiag').style.display=mode==='live'?'block':'none';$('#replayPane').style.display=mode==='live'?'none':'block';$('#seek').style.display=mode==='replay'?'':'none';$('#playback').style.display=mode==='replay'?'':'none';if(mode==='live'){$('#title').textContent=`Live WebTiles — agent ${policySlot+1}`;stop()}else if(mode==='train'){$('#title').textContent=`Training B / env ${trainEnv} — exact policy observation`}else{$('#title').textContent=frames.length?'Replay Lab — exact policy observation':'Replay Lab'}}
function paintCore(f,heading){const rows=(f.screen||f.state||'').split('\n'),cols=(f.colors||'').split('\n');let out='';for(let y=0;y<24;y++){const row=rows[y]||'',cr=cols[y]||'';for(let x=0;x<80;x++){out+=`<span class="${esc(cr[x]||'w')}">${esc(row[x]||' ')}</span>`}out+='\n'}$('#term').innerHTML=out;const names=f.names||[],ps=f.probs||[];let d=`<b>${heading}</b><p><span class="action">${esc(f.action||'?')}</span> · ${esc(f.t??'?')}s · value ${esc(f.value??'?')}</p>`;if(ps.length){d+='<b>Action probabilities</b><div class="probgrid">'+ps.map((p,n)=>`<div class="prob ${names[n]===f.action?'picked':''}"><span>${esc(names[n]||n)}</span><span class="track"><span class="fill" style="width:${Math.max(2,p/Math.max(...ps)*100)}%"></span></span><span>${(p*100).toFixed(0)}%</span></div>`).join('')+'</div>'}$('#diag').innerHTML=d;}
function paint(f){paintCore(f,`Frame ${i+1}/${frames.length}`)}
function show(n){if(!frames.length)return;i=Math.max(0,Math.min(n,frames.length-1));$('#seek').value=i;paint(frames[i]);}
function stop(){if(timer){clearInterval(timer);timer=null}$('#play').textContent='Play'}function toggle(){if(timer){stop();return}timer=setInterval(()=>{if(i>=frames.length-1)return stop();show(i+1)},+$('#speed').value);$('#play').textContent='Pause'}
async function openReplay(id){frames=await (await fetch('/api/replay?id='+encodeURIComponent(id))).json();setMode('replay');$('#title').textContent=`Replay Lab — ${id}`;$('#seek').max=Math.max(0,frames.length-1);show(0);}
function liveProbabilities(s){const names=s.action_names||[],ps=s.action_probabilities||[];if(!ps.length){$('#liveDiag').innerHTML='<span class="muted">Waiting for this neural agent’s first decision…</span>';return}const peak=Math.max(...ps,0.0001);$('#liveDiag').innerHTML=`<b>Agent ${s.slot+1} action probabilities</b><span class="muted"> · chose </span><span class="action">${esc(s.last_action||'?')}</span><div class="probgrid">${ps.map((p,n)=>`<div class="prob ${names[n]===s.last_action?'picked':''}"><span>${esc(names[n]||n)}</span><span class="track"><span class="fill" style="width:${Math.max(2,p/peak*100)}%"></span></span><span>${(p*100).toFixed(1)}%</span></div>`).join('')}</div>`;}
function selectPolicy(slot){policySlot=slot;setMode('live');$('#webtiles').src=location.origin+'/live?slot='+slot;policyStatus();}
async function policyStatus(){const all=await (await fetch('/api/policy/status',{cache:'no-store'})).json(),b=$('#policyBadge');policyAgents=all.agents||[];const s=policyAgents[policySlot]||{slot:policySlot,phase:'offline'};b.textContent=`${all.running_count||0}/${all.agent_count||2} ${all.phase||'offline'}`;b.className='badge '+(all.phase||'');$('#pd').textContent=s.depth?`D:${s.depth}`:'—';$('#php').textContent=s.hp_max?`${s.hp}/${s.hp_max}`:'—';$('#pt').textContent=s.turn??'—';const run=`attempt ${s.attempt||1} · best D:${s.best_depth||s.depth||1}`;const checkpoint=`${s.lane||'fixed'} · ${s.checkpoint_architecture||'unknown'} · update ${s.checkpoint_update||0} · ${(s.checkpoint_sha256||'').slice(0,8)}`;let msg=s.outcome||s.last_action||'Ready for a checkpoint canary run.';if(s.phase==='validating')msg=`10-minute live gate · ${s.validation_remaining_s}s · ${run} · ${s.actions||0} neural actions`;else if(s.phase==='running')msg=`Validation passed · ${run} · ${s.actions||0} neural actions · ${s.last_action||''}`;$('#policyText').textContent=`${checkpoint} · ${msg}`;const grid=$('#policyGrid');grid.innerHTML=policyAgents.map(a=>`<button class="policycell ${a.slot===policySlot?'on':''}" data-slot="${a.slot}"><b>${a.lane||'#'+(a.slot+1)} · ${a.depth?'D:'+a.depth:a.phase||'offline'}</b><small>${a.running?'turn '+(a.turn||0):a.phase||'offline'} · ${a.checkpoint_architecture||'?'} · u${a.checkpoint_update||0}</small></button>`).join('');grid.querySelectorAll('.policycell').forEach(cell=>cell.onclick=()=>selectPolicy(+cell.dataset.slot));$('#startPolicy').disabled=!!all.running;$('#stopPolicy').disabled=!all.running;$('#policyReplay').disabled=!s.replay;$('#policyReplay').dataset.replay=s.replay||'';liveProbabilities(s);}
async function policyPost(path){await fetch(path,{method:'POST'});await policyStatus();}
async function pollEnvs(){let list=[];try{list=await (await fetch('/api/training/envs?v=c',{cache:'no-store'})).json()}catch(e){}const g=$('#envGrid');if(!list.length){g.innerHTML='<span class="muted">no C environments running</span>';return}g.innerHTML=list.map(e=>{const hp=Math.round(100*(e.hp??1));return `<button class="envcell ${e.env===trainEnv?'on':''}" data-env="${e.env}"><b>#${e.env} · D:${e.depth}</b><small>XL ${e.xl} · turn ${e.turns}</small><span class="hpbar"><i class="${hp<40?'low':''}" style="width:${hp}%"></i></span></button>`}).join('');g.querySelectorAll('.envcell').forEach(b=>b.onclick=async()=>{trainEnv=+b.dataset.env;setMode('train');$('#envMeta').textContent=`switching to C / env ${trainEnv}…`;await fetch(`/api/training/watch?v=c&env=${trainEnv}`,{method:'POST'});pollEnvs();pollTraining()})}
async function pollTraining(){if(mode!=='train')return;try{const f=await (await fetch('/api/training/live?v=c',{cache:'no-store'})).json();if(!f)return;if(f.env!==trainEnv){$('#envMeta').textContent=`switching to C / env ${trainEnv}…`;return}trainFrame=f;$('#envMeta').textContent=`C / env ${f.env} · step ${f.step} · ${f.action}`;$('#title').textContent=`Training C / env ${f.env} — exact recurrent policy observation`;paintCore(f,`Live C environment ${f.env} · step ${f.step}`)}catch(e){}}
$('#webtiles').src=location.origin+'/live?slot=0';$('#live').onclick=()=>selectPolicy(policySlot);$('#openLive').onclick=()=>window.open(location.origin+'/live?slot='+policySlot,'_blank');$('#tileReplay').onclick=()=>window.open('http://127.0.0.1:8102','_blank');$('#replay').onclick=()=>setMode('replay');$('#play').onclick=toggle;$('#seek').oninput=e=>show(+e.target.value);loadGames();
$('#startPolicy').onclick=()=>policyPost('/api/policy/start');$('#stopPolicy').onclick=()=>policyPost('/api/policy/stop');$('#policyReplay').onclick=e=>window.open('http://127.0.0.1:8102/?id='+encodeURIComponent(e.target.dataset.replay),'_blank');policyStatus();pollEnvs();pollTraining();setInterval(policyStatus,1000);setInterval(pollEnvs,1000);setInterval(pollTraining,500);
</script>"""


def replay_files(data_root):
    return sorted(data_root.glob("rl_replays_*/*.jsonl"),
                  key=lambda p: p.stat().st_mtime, reverse=True)


def load_frames(data_root, replay_id):
    # IDs are compared to file stems, never interpreted as paths.
    for path in replay_files(data_root):
        if path.stem == replay_id:
            out = []
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
            return out
    return []


def app(data_root, webtiles_url):
    runtime = {"processes": {}, "logs": {}}
    checkpoint = Path("/mnt/c/Users/jorda/dcss-gym/data/rl_policy.c16.gym.pt")
    for lane in POLICY_LANES:
        lane_manifest = manifest_path(data_root, "c", lane)
        if not lane_manifest.exists() and checkpoint.exists():
            publish_manifest(
                checkpoint, lane_manifest, variant="c", channel=lane,
                architecture="spatial-v1",
                action_names=("autofight", "explore", "rest", "descend",
                              "travel", "escape", "berserk", "move_n",
                              "move_ne", "move_e", "move_se", "move_s",
                              "move_sw", "move_w", "move_nw", "wait"),
                metrics={"bootstrap": "gym-certified"})

    def external_policy_pids():
        """Adopt evaluators that survived a dashboard-only restart."""
        found = {}
        for proc in Path("/proc").glob("[0-9]*"):
            try:
                command = (proc / "cmdline").read_bytes().split(b"\0")
            except OSError:
                continue
            if not any(b"webtiles_policy_agent.py" in part
                       for part in command):
                continue
            slot = 0
            try:
                marker = command.index(b"--slot")
                slot = int(command[marker + 1])
            except (ValueError, IndexError):
                pass
            if 0 <= slot < POLICY_AGENTS:
                found[slot] = int(proc.name)
        return found

    def policy_status():
        external = external_policy_pids()
        agents = []
        for slot, username in enumerate(POLICY_USERS):
            process = runtime["processes"].get(slot)
            pid = (process.pid if process and process.poll() is None
                   else external.get(slot))
            running = pid is not None
            path = data_root / f"webtiles_policy_live.{slot}.json"
            try:
                status = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                status = {"phase": "offline"}
            status.update({
                "running": running, "pid": pid, "slot": slot,
                "username": username, "lane": POLICY_LANES[slot],
            })
            if (not running
                    and status.get("phase") in {"validating", "running"}):
                status["phase"] = "failed"
                status.setdefault("outcome", "policy process stopped")
            agents.append(status)
        active = [agent for agent in agents if agent["running"]]
        return {
            "agents": agents, "running": bool(active),
            "agent_count": POLICY_AGENTS,
            "running_count": len(active),
            "phase": ("validating" if any(
                agent.get("phase") == "validating" for agent in active)
                else "running" if active else "offline"),
        }

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def send(self, code, body, content_type="text/plain; charset=utf-8"):
            blob = body.encode("utf-8") if isinstance(body, str) else body
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(blob)))
            self.end_headers()
            self.wfile.write(blob)

        def do_GET(self):
            u = urlparse(self.path)
            if u.path == "/":
                return self.send(200, PAGE, "text/html; charset=utf-8")
            if u.path == "/live":
                # Browser-level redirect keeps the live renderer official rather
                # than reverse-proxying WebTiles sockets or authentication.
                self.send_response(302)
                try:
                    slot = max(0, min(POLICY_AGENTS - 1, int(
                        parse_qs(u.query).get("slot", ["0"])[0])))
                except ValueError:
                    slot = 0
                self.send_header(
                    "Location", f"{webtiles_url}/#watch-{POLICY_USERS[slot]}")
                self.end_headers()
                return
            if u.path == "/api/replays":
                rows = []
                for p in replay_files(data_root):
                    try:
                        first = json.loads(p.open(encoding="utf-8").readline())
                    except (OSError, json.JSONDecodeError):
                        first = {}
                    rows.append({"id": p.stem, "frames": sum(1 for _ in p.open(encoding="utf-8")),
                                 "variant": first.get("variant", "")})
                return self.send(200, json.dumps(rows), "application/json")
            if u.path == "/api/replay":
                rid = parse_qs(u.query).get("id", [""])[0]
                return self.send(200, json.dumps(load_frames(data_root, rid)), "application/json")
            if u.path == "/api/policy/status":
                return self.send(200, json.dumps(policy_status()), "application/json")
            if u.path == "/api/training/envs":
                variant = parse_qs(u.query).get("v", ["b"])[0]
                variant = variant if variant in {"a", "b", "c"} else "b"
                try:
                    body = (data_root / f"rl_envs.{variant}.json").read_text("utf-8")
                except OSError:
                    body = "[]"
                return self.send(200, body, "application/json")
            if u.path == "/api/training/live":
                variant = parse_qs(u.query).get("v", ["b"])[0]
                variant = variant if variant in {"a", "b", "c"} else "b"
                try:
                    body = (data_root / f"rl_live.{variant}.json").read_text("utf-8")
                except OSError:
                    body = "null"
                return self.send(200, body, "application/json")
            return self.send(404, "not found")

        def do_POST(self):
            # Phone sharing is deliberately spectator-only. Localhost retains
            # the start/stop and training controls, but a public quick-tunnel
            # request can never mutate agent state even if someone discovers
            # its temporary random hostname.
            if is_tunnel_request(self.headers):
                return self.send(
                    403, json.dumps({"error": "remote dashboard is read-only"}),
                    "application/json")
            u = urlparse(self.path)
            if u.path == "/api/policy/start":
                if policy_status().get("running"):
                    return self.send(409, json.dumps({"error": "already running"}), "application/json")
                started = []
                for slot, username in enumerate(POLICY_USERS):
                    log = (data_root / f"webtiles_policy.{slot}.log").open("ab")
                    runtime["logs"][slot] = log
                    command = [
                        sys.executable, "-u",
                        str(HERE / "webtiles_policy_agent.py"),
                        "--checkpoint-manifest", str(manifest_path(
                            data_root, "c", POLICY_LANES[slot])),
                        "--variant", "c",
                        "--fresh", "--repeat", "--register",
                        "--username", username, "--password", "midca",
                        "--slot", str(slot), "--live-file",
                        str(data_root / f"webtiles_policy_live.{slot}.json"),
                        "--validation-minutes", "10", "--stall-timeout", "20",
                        "--retry", "1",
                    ]
                    if POLICY_LANES[slot] == "champion":
                        # The promoted lane is an evaluation canary, not an
                        # exploration actor; make its replay reproducible.
                        command.append("--deterministic")
                    process = subprocess.Popen(
                        command, cwd=HERE, stdout=log,
                        stderr=subprocess.STDOUT)
                    runtime["processes"][slot] = process
                    started.append({"slot": slot, "pid": process.pid})
                return self.send(202, json.dumps({
                    "started": True, "agents": started,
                }), "application/json")
            if u.path == "/api/policy/stop":
                pids = set(external_policy_pids().values())
                for process in runtime["processes"].values():
                    if process.poll() is None:
                        process.terminate()
                        pids.discard(process.pid)
                for process in runtime["processes"].values():
                    if process.poll() is None:
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            process.kill()
                for pid in pids:
                    os.kill(pid, signal.SIGTERM)
                return self.send(200, json.dumps({
                    "stopped": True, "agents": POLICY_AGENTS,
                }), "application/json")
            if u.path == "/api/training/watch":
                q = parse_qs(u.query)
                variant = q.get("v", ["b"])[0]
                variant = variant if variant in {"a", "b", "c"} else "b"
                try:
                    env = max(0, min(255, int(q.get("env", ["0"])[0])))
                    (data_root / f"rl_view.{variant}.txt").write_text(str(env), "utf-8")
                except (OSError, ValueError) as exc:
                    return self.send(400, json.dumps({"error": str(exc)}), "application/json")
                return self.send(200, json.dumps({"variant": variant, "env": env}), "application/json")
            return self.send(404, "not found")
    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8101)
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--webtiles", default="http://127.0.0.1:8090")
    args = ap.parse_args()
    print(f"DCSS Spectator: http://127.0.0.1:{args.port}")
    ThreadingHTTPServer(("127.0.0.1", args.port), app(args.data_root, args.webtiles)).serve_forever()


if __name__ == "__main__":
    main()
