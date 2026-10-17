# Collects game records that players of the public page sent in as GitHub issues ("[宗师棋谱] ..."): wins against
# 宗师 only. Issues are public input: records are size-limited, parsed as data, and human_samples.js later replays
# every move under the real rules and drops records that are illegal or whose result does not match.
#   python3 train/az/collect_online.py          (keeps running; checks every 10 minutes)
import base64, json, os, re, subprocess, time, zlib

HERE = os.path.dirname(os.path.abspath(__file__)); GAMES = os.path.join(HERE, 'games')
SEEN = os.path.join(HERE, 'out', 'online_seen.json'); LOG = os.path.join(HERE, 'out', 'collect_log.txt')
API = 'https://api.github.com/repos/pphy96826-coder/hexwar/issues?state=all&per_page=100&sort=created&direction=desc'
POLL = 600

def log(m):
    line = time.strftime('[%m-%d %H:%M:%S] ') + m; print(line, flush=True); open(LOG, 'a').write(line + '\n')

def decode(body):
    m = re.search(r'```hexwar-record\s*([A-Za-z0-9_\-]{20,200000})\s*```', body or '')
    if not m: return None
    code = m.group(1); raw = base64.urlsafe_b64decode(code + '=' * (-len(code) % 4))
    data = zlib.decompressobj(-15).decompress(raw, 2_000_000)
    return json.loads(data)

def valid_win(g):
    if not isinstance(g, dict) or g.get('game') != '蜂巢兵棋' or not isinstance(g.get('moves'), list) or len(g['moves']) > 2000: return False
    sides, levels = g.get('aiSides'), g.get('aiLevels')
    if not (isinstance(sides, list) and isinstance(levels, list) and len(sides) == 2 and sides.count(True) == 1): return False
    ai = sides.index(True); human = 1 - ai
    return levels[ai] == 'neural' and g.get('winner') == human

def once(seen):
    # the system curl (this Python build has no CA certificates of its own)
    out = subprocess.run(['curl', '-sfL', '--max-time', '30', '-H', 'Accept: application/vnd.github+json', '-H', 'User-Agent: hexwar-collector', API],
                         capture_output=True, check=True).stdout
    issues = json.loads(out)
    new = 0
    for it in issues:
        n = it.get('number'); title = it.get('title') or ''
        if 'pull_request' in it or str(n) in seen or not title.startswith('[宗师棋谱]'): continue
        seen[str(n)] = 'bad'
        try:
            g = decode(it.get('body'))
            if g and valid_win(g):
                g['source'] = f'github issue #{n}'
                with open(os.path.join(GAMES, f'online_{n}.json'), 'w', encoding='utf8') as f: json.dump(g, f, ensure_ascii=False)
                os.utime(GAMES); seen[str(n)] = 'ok'; new += 1
                log(f'issue #{n}: saved ({g.get("map")}, {g.get("round")} rounds, network gen {g.get("nnGen")})')
            else: log(f'issue #{n}: not a valid win against 宗师, skipped')
        except Exception as e:
            log(f'issue #{n}: unreadable record ({e!r}), skipped')
    return new

def main():
    seen = json.load(open(SEEN)) if os.path.exists(SEEN) else {}
    while True:
        try:
            if once(seen): pass
            json.dump(seen, open(SEEN, 'w'))
        except Exception as e:
            log(f'check failed: {e!r}')
        time.sleep(POLL)

if __name__ == '__main__':
    main()
