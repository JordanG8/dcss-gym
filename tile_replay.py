"""Play recorded WebTiles protocol streams through DCSS's official client.

Unlike terminal JSONL replays, these recordings contain the exact `map` and
`player` messages (including server-assigned tile IDs) that WebTiles sent while
the game ran. The page below reuses the local DCSS WebTiles JavaScript and
sprite assets; it only replaces the socket with a read-only recorded stream.

    /root/pty-venv/bin/python tile_replay.py
    # http://127.0.0.1:8102

Start a recordable watchable run with:
    python attic/webtiles_agent.py --record-tiles
"""
import argparse
import json
import mimetypes
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from urllib.request import urlopen

from dcss_env import CRAWL_DIR


HERE = Path(__file__).parent
REPLAYS = HERE / "data" / "webtiles_replays"
WEB_ROOT = Path(CRAWL_DIR) / "webserver"
COMMON_STATIC = WEB_ROOT / "static"
GAME_STATIC = WEB_ROOT / "game_data" / "static"


# Injected *before* the official RequireJS client starts. It behaves like a
# WebSocket just enough for the existing client to initialize normally, then
# feeds original messages back through its ordinary `onmessage` handler.
REPLAY_SHIM = r"""
<style>
#tile-replay-controls{position:fixed;z-index:10000;right:12px;bottom:12px;
 background:#101713eF;color:#d9ddd5;border:1px solid #52684f;border-radius:8px;
 padding:9px;font:13px system-ui,sans-serif;box-shadow:0 4px 20px #000}
#tile-replay-controls button{background:#203321;color:#e8f4dc;border:1px solid #6c9861;
 border-radius:4px;padding:5px 8px;margin-right:5px;cursor:pointer}
#tile-replay-status{display:block;margin-top:6px;color:#b6c9ac}
</style>
<script>
(function(){
  const replayId = new URLSearchParams(location.search).get('id') || '';
  let sock, events=[], cursor=0, timer=null, readyResolve;
  window.__tileReplayReady = new Promise(r => readyResolve=r);
  window.WebSocket = function ReplaySocket(){
    sock=this; this.readyState=0; this.binaryType='arraybuffer'; this.send=()=>{};
    setTimeout(()=>{this.readyState=1; if(this.onopen)this.onopen({}); readyResolve();},0);
  };
  window.WebSocket.prototype.close=function(){this.readyState=3;if(this.onclose)this.onclose({});};
  function emit(e){ if(sock && sock.onmessage) sock.onmessage({data:JSON.stringify(e.data)}); }
  function sendUntil(target){
    while(cursor<events.length && (events[cursor].t||0)<=target) emit(events[cursor++]);
    status();
  }
  function status(){ const el=document.querySelector('#tile-replay-status');
    if(el) el.textContent='Recorded turn '+Math.min(cursor?events[cursor-1].t||0:0, maxTurn())+' · '+cursor+'/'+events.length+' messages'; }
  function maxTurn(){ return events.length ? (events[events.length-1].t||0) : 0; }
  function next(){ const turn=cursor<events.length ? (events[cursor].t||0) : maxTurn(); sendUntil(turn); }
  function play(){ if(timer){clearInterval(timer);timer=null;return status();}
    timer=setInterval(()=>{if(cursor>=events.length){clearInterval(timer);timer=null;return status();} next();},250); status(); }
  async function boot(){
    events=await (await fetch('/api/tile-replay?id='+encodeURIComponent(replayId))).json();
    await window.__tileReplayReady;
    // Deliver the authentic login/game-client setup first, then give the
    // official RequireJS renderer a moment to initialize before its first map
    // arrives. The player lands immediately on a genuine tiled game state.
    sendUntil(0);
    await new Promise(r=>setTimeout(r,700));
    // The first map is a partial clear; use the first follow-up delta when it
    // exists so the canvas opens on the same complete tile state a spectator
    // would see after WebTiles has settled.
    const maps=events.filter(e=>e.data && e.data.msg==='map');
    const firstMap=maps[Math.min(1,maps.length-1)];
    if(firstMap) sendUntil(firstMap.t||0);
    const box=document.createElement('div'); box.id='tile-replay-controls';
    box.innerHTML='<select id="tr-list" aria-label="Recorded tiled replay"></select><br><button id="tr-next">Next turn</button><button id="tr-play">Play</button><button id="tr-reset">Restart</button><span id="tile-replay-status"></span>';
    document.body.appendChild(box);
    const list=await (await fetch('/api/tile-replays')).json();
    const select=document.querySelector('#tr-list');
    for(const item of list){const o=document.createElement('option');o.value=item.id;o.textContent=item.id+' · '+item.turns+' turns';o.selected=item.id===replayId;select.appendChild(o)}
    select.onchange=()=>location.href='/?id='+encodeURIComponent(select.value);
    document.querySelector('#tr-next').onclick=next;
    document.querySelector('#tr-play').onclick=function(){play();this.textContent=timer?'Pause':'Play';};
    document.querySelector('#tr-reset').onclick=()=>location.reload();
    status();
  }
  boot().catch(e=>{document.body.insertAdjacentHTML('beforeend','<pre>Tile replay failed: '+String(e)+'</pre>');});
})();
</script>
"""


def replay_path(replay_id):
    """Resolve only a known file stem; query strings are never filesystem paths."""
    for path in REPLAYS.glob("*.json"):
        if path.stem == replay_id:
            return path
    return None


def replay_events(replay_id):
    path = replay_path(replay_id)
    if path is None:
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("format") != "dcss-webtiles-stream-v1":
        return []
    return data.get("events", [])


def replay_index():
    rows = []
    for path in sorted(REPLAYS.glob("*.json"), reverse=True):
        try:
            events = json.loads(path.read_text(encoding="utf-8")).get("events", [])
            rows.append({"id": path.stem,
                         "turns": max((event.get("t", 0) for event in events), default=0),
                         "messages": len(events)})
        except (OSError, json.JSONDecodeError):
            continue
    return rows


def client_shell():
    """Get DCSS's rendered client page and add the read-only socket shim."""
    with urlopen("http://127.0.0.1:8090/", timeout=3) as response:
        page = response.read().decode("utf-8", "replace")
    # `require.js` executes while HTML is parsed, so the shim must be placed
    # immediately before it rather than appended at the end of the document.
    pattern = r'(<script[^>]+require\.js[^>]*></script>)'
    page, count = re.subn(pattern, REPLAY_SHIM + r"\1", page, count=1)
    if count != 1:
        raise RuntimeError("could not locate the official WebTiles bootstrap")
    return page


def static_file(url_path):
    relative = Path(url_path.removeprefix("/static/"))
    if relative.is_absolute() or ".." in relative.parts:
        return None
    for root in (COMMON_STATIC, GAME_STATIC):
        candidate = root / relative
        if candidate.is_file():
            return candidate
    return None


def game_data_file(url_path):
    """Resolve /gamedata/<recorded-version>/<asset> to local Crawl assets."""
    parts = Path(url_path.removeprefix("/gamedata/")).parts
    if len(parts) < 2 or ".." in parts:
        return None
    candidate = GAME_STATIC.joinpath(*parts[1:])
    return candidate if candidate.is_file() else None


def app():
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def send(self, status, body, content_type="text/plain; charset=utf-8"):
            blob = body.encode("utf-8") if isinstance(body, str) else body
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(blob)))
            self.end_headers()
            self.wfile.write(blob)

        def do_GET(self):
            request = urlparse(self.path)
            if request.path == "/":
                ids = sorted((p.stem for p in REPLAYS.glob("*.json")), reverse=True)
                if not ids:
                    return self.send(404, "No native WebTiles recordings yet. Run attic/webtiles_agent.py --record-tiles.")
                # Default to the newest stream, while the select-free URL
                # remains shareable and deterministic.
                replay_id = parse_qs(request.query).get("id", [ids[0]])[0]
                if replay_path(replay_id) is None:
                    return self.send(404, "unknown tiled replay")
                if "id" not in parse_qs(request.query):
                    self.send_response(302)
                    self.send_header("Location", f"/?id={replay_id}")
                    self.end_headers()
                    return
                try:
                    return self.send(200, client_shell(), "text/html; charset=utf-8")
                except Exception as exc:
                    return self.send(503, f"WebTiles client unavailable: {exc}")
            if request.path == "/api/tile-replay":
                replay_id = parse_qs(request.query).get("id", [""])[0]
                return self.send(200, json.dumps(replay_events(replay_id)), "application/json")
            if request.path == "/api/tile-replays":
                return self.send(200, json.dumps(replay_index()), "application/json")
            if request.path.startswith("/static/"):
                path = static_file(request.path)
                if path is None:
                    return self.send(404, "asset not found")
                return self.send(200, path.read_bytes(),
                                 mimetypes.guess_type(path.name)[0] or "application/octet-stream")
            if request.path.startswith("/gamedata/"):
                path = game_data_file(request.path)
                if path is None:
                    return self.send(404, "game asset not found")
                return self.send(200, path.read_bytes(),
                                 mimetypes.guess_type(path.name)[0] or "application/octet-stream")
            return self.send(404, "not found")
    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8102)
    args = ap.parse_args()
    print(f"Native WebTiles replay: http://127.0.0.1:{args.port}")
    ThreadingHTTPServer(("127.0.0.1", args.port), app()).serve_forever()


if __name__ == "__main__":
    main()
