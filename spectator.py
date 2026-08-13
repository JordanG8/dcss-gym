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
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


HERE = Path(__file__).parent
DEFAULT_DATA = HERE / "data"

PAGE = r"""<!doctype html><meta charset="utf-8"><title>DCSS Spectator</title>
<style>
:root{color-scheme:dark;--ink:#d9ddd5;--muted:#92a098;--edge:#29362e;--pane:#101713;--accent:#9fd36b;--hot:#f4c66a}*{box-sizing:border-box}body{margin:0;background:#080d0a;color:var(--ink);font:14px system-ui,sans-serif}header{height:56px;display:flex;align-items:center;gap:14px;padding:0 22px;border-bottom:1px solid var(--edge);background:#101713}h1{margin:0;font-size:17px;letter-spacing:.4px}h1 b{color:var(--accent)}button,select,input{background:#18231d;color:var(--ink);border:1px solid #35473b;border-radius:6px;padding:7px 10px}button:hover{border-color:var(--accent);cursor:pointer}.wrap{display:grid;grid-template-columns:minmax(560px,1fr) 340px;gap:14px;padding:14px;max-width:1700px;margin:auto}.pane{border:1px solid var(--edge);border-radius:10px;background:var(--pane);overflow:hidden}.bar{padding:10px 12px;border-bottom:1px solid var(--edge);display:flex;gap:8px;align-items:center;flex-wrap:wrap}.bar .spacer{flex:1}.live{width:100%;height:calc(100vh - 160px);min-height:610px;border:0;background:#000}.replay{display:none}.term{margin:0;padding:14px;background:#020403;min-height:576px;overflow:auto;font:15px/1.12 ui-monospace,Consolas,monospace;white-space:pre;letter-spacing:0}.term span{white-space:pre}.w{color:#d7ddd4}.r{color:#ec6b6b}.g{color:#83d17e}.y{color:#d6b565}.b{color:#80a6ef}.m{color:#d987d9}.c{color:#71d7d5}.k{color:#424d44}.W{color:#fff}.R{color:#ff8b8b}.G{color:#aafa96}.Y{color:#ffe08b}.B{color:#a7c5ff}.M{color:#f2a5f1}.C{color:#9af4ef}.side{padding:12px}.muted{color:var(--muted)}.game{width:100%;text-align:left;margin:4px 0}.game small{display:block;color:var(--muted);margin-top:2px}.action{color:var(--hot);font-weight:700}.prob{display:grid;grid-template-columns:78px 1fr 38px;gap:7px;align-items:center;margin:5px 0;font-size:12px}.track{height:7px;background:#26352c;border-radius:9px;overflow:hidden}.fill{height:100%;background:#81b857}.picked .fill{background:var(--hot)}#diag{margin-top:12px;border-top:1px solid var(--edge);padding-top:10px}@media(max-width:1000px){.wrap{grid-template-columns:1fr}.live{height:65vh}.side{max-height:40vh;overflow:auto}}
</style><header><h1><b>DCSS</b> Spectator</h1><button id="live">Live WebTiles</button><button id="openLive">Open WebTiles</button><button id="replay">Replay Lab</button><span class="muted">player-visible replays · local-only director diagnostics</span></header>
<main class="wrap"><section class="pane"><div class="bar"><b id="title">Live game — official WebTiles renderer</b><span class="spacer"></span><button id="play">Play</button><label>speed <select id="speed"><option value="900">slow</option><option value="350" selected>normal</option><option value="100">fast</option></select></label></div><iframe id="webtiles" class="live"></iframe><div id="replayPane" class="replay"><pre id="term" class="term"></pre><input id="seek" type="range" min="0" value="0" style="width:100%"></div></section><aside class="pane side"><b>Recorded episodes</b><p class="muted">Live play is the authentic WebTiles page. Replays show exactly what the policy saw, including terminal colours and action probabilities.</p><div id="games"></div><div id="diag"></div></aside></main>
<script>
const $=q=>document.querySelector(q);let frames=[],i=0,timer=null,mode='live';
function esc(s){return String(s??'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))}
async function loadGames(){const gs=await (await fetch('/api/replays')).json();$('#games').innerHTML=gs.length?gs.map(g=>`<button class="game" data-id="${esc(g.id)}">${esc(g.id)}<small>${g.frames} frames · ${esc(g.variant||'legacy')}</small></button>`).join(''):'<p class="muted">No replay files found in this data folder.</p>';document.querySelectorAll('.game').forEach(b=>b.onclick=()=>openReplay(b.dataset.id));}
function setMode(next){mode=next;$('#webtiles').style.display=mode==='live'?'block':'none';$('#replayPane').style.display=mode==='replay'?'block':'none';$('#title').textContent=mode==='live'?'Live game — official WebTiles renderer':frames.length?'Replay Lab — exact policy observation':'Replay Lab';if(mode==='live')stop();}
function paint(f){const rows=(f.screen||f.state||'').split('\n'),cols=(f.colors||'').split('\n');let out='';for(let y=0;y<24;y++){const row=rows[y]||'',cr=cols[y]||'';for(let x=0;x<80;x++){out+=`<span class="${esc(cr[x]||'w')}">${esc(row[x]||' ')}</span>`}out+='\n'}$('#term').innerHTML=out;const names=f.names||[],ps=f.probs||[];let d=`<b>Frame ${i+1}/${frames.length}</b><p><span class="action">${esc(f.action||'?')}</span> · ${esc(f.t??'?')}s · value ${esc(f.value??'?')}</p>`;if(ps.length){d+='<b>Policy distribution</b>'+ps.map((p,n)=>`<div class="prob ${names[n]===f.action?'picked':''}"><span>${esc(names[n]||n)}</span><span class="track"><span class="fill" style="width:${Math.max(2,p/Math.max(...ps)*100)}%"></span></span><span>${(p*100).toFixed(0)}%</span></div>`).join('')}$('#diag').innerHTML=d;}
function show(n){if(!frames.length)return;i=Math.max(0,Math.min(n,frames.length-1));$('#seek').value=i;paint(frames[i]);}
function stop(){if(timer){clearInterval(timer);timer=null}$('#play').textContent='Play'}function toggle(){if(timer){stop();return}timer=setInterval(()=>{if(i>=frames.length-1)return stop();show(i+1)},+$('#speed').value);$('#play').textContent='Pause'}
async function openReplay(id){frames=await (await fetch('/api/replay?id='+encodeURIComponent(id))).json();setMode('replay');$('#title').textContent=`Replay Lab — ${id}`;$('#seek').max=Math.max(0,frames.length-1);show(0);}
$('#webtiles').src=location.origin+'/live';$('#live').onclick=()=>setMode('live');$('#openLive').onclick=()=>window.open(location.origin+'/live','_blank');$('#replay').onclick=()=>setMode('replay');$('#play').onclick=toggle;$('#seek').oninput=e=>show(+e.target.value);loadGames();
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
                self.send_header("Location", webtiles_url)
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
