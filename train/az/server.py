# Local server for playing against the network while it trains.
#   python3 train/az/server.py [port]        (serves the project folder; open http://localhost:8765/hexwar.html)
# GET  /api/net       latest promoted network for the page (train/az/web/net_best.bin, written by train.py)
# GET  /api/net-info  {"gen": n} of that network
# POST /api/game      a finished game record (JSON) -> train/az/games/web_<time>.json, used by the next generation
import json, os, sys, time
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
WEB = os.path.join(HERE, 'web'); GAMES = os.path.join(HERE, 'games')

class H(SimpleHTTPRequestHandler):
    def __init__(self, *a, **k): super().__init__(*a, directory=ROOT, **k)
    def log_message(self, fmt, *a): pass
    def send_json(self, code, obj):
        b = json.dumps(obj).encode(); self.send_response(code)
        self.send_header('Content-Type', 'application/json'); self.send_header('Content-Length', str(len(b)))
        self.send_header('Cache-Control', 'no-store'); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        if self.path.startswith('/api/net-info'):
            p = os.path.join(WEB, 'net_best.json')
            return self.send_json(200, json.load(open(p)) if os.path.exists(p) else {'gen': None})
        if self.path.startswith('/api/net'):
            p = os.path.join(WEB, 'net_best.bin')
            if not os.path.exists(p): return self.send_json(404, {'error': 'no network yet'})
            b = open(p, 'rb').read(); self.send_response(200)
            self.send_header('Content-Type', 'application/octet-stream'); self.send_header('Content-Length', str(len(b)))
            self.send_header('Cache-Control', 'no-store'); self.end_headers(); self.wfile.write(b); return
        return super().do_GET()
    def do_POST(self):
        if not self.path.startswith('/api/game'): return self.send_json(404, {'error': 'not found'})
        n = int(self.headers.get('Content-Length') or 0)
        if n <= 0 or n > 2_000_000: return self.send_json(400, {'error': 'bad size'})
        try:
            g = json.loads(self.rfile.read(n))
            assert g.get('game') == '蜂巢兵棋' and isinstance(g.get('moves'), list) and g.get('winner') is not None
            assert isinstance(g.get('aiSides'), list) and not all(g['aiSides'])   # a person played at least one side
        except Exception as e:
            return self.send_json(400, {'error': 'not a finished game record'})
        os.makedirs(GAMES, exist_ok=True)
        name = time.strftime('web_%Y%m%d_%H%M%S') + f'_{int(time.time()*1000)%1000:03d}.json'
        with open(os.path.join(GAMES, name), 'w', encoding='utf8') as f: json.dump(g, f, ensure_ascii=False)
        os.utime(GAMES)   # the trainer reloads player games when this folder changes
        return self.send_json(200, {'saved': name})

if __name__ == '__main__':
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    print(f'serving {ROOT} on http://localhost:{port}', flush=True)
    ThreadingHTTPServer(('127.0.0.1', port), H).serve_forever()
