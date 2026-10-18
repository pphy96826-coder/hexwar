# Publishes each newly promoted network to GitHub (code on main, the playable page on gh-pages = GitHub Pages).
#   python3 train/az/deploy_github.py            (keeps running; checks every few minutes)
# A generation is promoted by train.py only after beating the previous best (>= 55% over 40 games). Before it goes
# online it must also (1) score >= 50% against the version online now and >= 50% on average against the last POOL
# online versions (60 games each, network only, 200 searches per move; random maps, sampled openings and random picks
# among near-equal moves, so it cannot pass by learning one opponent's fixed lines) and (2) still beat the
# hand-written master >= 90% (48 games). At most one push every MIN_GAP seconds. State in train/az/out/deploy_state.json.
import json, os, re, shutil, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
REPO = os.path.join(ROOT, 'hexwar-github'); WEB = os.path.join(HERE, 'web')
MIN_GAP = 3 * 3600; POLL = 300
LOG = os.path.join(HERE, 'out', 'deploy_log.txt'); STATE = os.path.join(HERE, 'out', 'deploy_state.json')
TEST_GAMES = 48; H2H_GAMES = 60; MASTER_FLOOR = 0.90; POOL = 3
CODE = {   # project file -> repository path (kept in sync on every publish)
    'hexwar_base.html': 'hexwar_base.html',
    **{f'train/az/{f}': f'train/az/{f}' for f in ['train.py', 'model.py', 'selfplay.js', 'game.js', 'mcts.js', 'maps.js', 'human_samples.js',
                                                 'eval.py', 'puzzle.py', 'export_web.py', 'server.py', 'run_forever.sh', 'deploy_github.py', 'collect_online.py', 'h2h.py', 'map_balance.py']},
    **{f'train/az/web/{f}': f'train/az/web/{f}' for f in ['nn.js', 'fastlegal.js', 'gameweb.js', 'mctsweb.js', 'threat.js', 'aiworker.js', 'build.py']},
}

def log(m):
    line = time.strftime('[%m-%d %H:%M:%S] ') + m; print(line, flush=True); open(LOG, 'a').write(line + '\n')

def git(*a, check=True):
    r = subprocess.run(['git', '-C', REPO, *a], capture_output=True, text=True)
    if check and r.returncode: raise RuntimeError(f"git {' '.join(a)}: {r.stderr.strip()[-400:]}")
    return r.stdout.strip()

def published_gen():
    m = re.search(r'当前最强的网络（第 (\d+) 代）', open(os.path.join(REPO, 'README.md'), encoding='utf8').read())
    return int(m.group(1)) if m else None

def publish(gen, why):
    tmp = os.path.join(REPO, 'hexwar.html.tmp')
    ck = os.path.join(HERE, 'out', f'gen{gen}.pt'); netbin = os.path.join(HERE, 'out', f'web_gen{gen}.bin')
    subprocess.run([sys.executable, os.path.join(HERE, 'export_web.py'), ck, netbin], check=True, capture_output=True)
    subprocess.run([sys.executable, os.path.join(WEB, 'build.py'), os.path.join(ROOT, 'hexwar_base.html'), netbin, tmp, '--gen', str(gen)],
                   check=True, capture_output=True)
    # the training engine must still load the page
    subprocess.run(['node', '-e', f"require({json.dumps(os.path.join(ROOT, 'train', 'engine'))})({json.dumps(tmp)})"], check=True, capture_output=True)
    os.replace(tmp, os.path.join(REPO, 'hexwar.html')); shutil.copyfile(os.path.join(REPO, 'hexwar.html'), os.path.join(REPO, 'index.html'))
    shutil.copyfile(netbin, os.path.join(REPO, 'models', 'net_best.bin')); shutil.copyfile(ck, os.path.join(REPO, 'models', 'best.pt'))
    for f in (netbin, netbin + '.json'):
        if os.path.exists(f): os.remove(f)
    for src, dst in CODE.items():
        if os.path.exists(os.path.join(ROOT, src)): shutil.copyfile(os.path.join(ROOT, src), os.path.join(REPO, dst))
    rd = os.path.join(REPO, 'README.md'); s = open(rd, encoding='utf8').read()
    s = re.sub(r'当前最强的网络（第 \d+ 代）', f'当前最强的网络（第 {gen} 代）', s)
    s = re.sub(r'（第 \d+ 代网络，随训练自动更新）', '', s)
    s = s.replace('- **在线试玩**：<https://pphy96826-coder.github.io/hexwar/>', f'- **在线试玩**：<https://pphy96826-coder.github.io/hexwar/>（第 {gen} 代网络，随训练自动更新）', 1)
    open(rd, 'w', encoding='utf8').write(s)
    git('add', '-A')
    if not git('status', '--porcelain'): log(f'gen {gen}: nothing changed'); return True
    git('-c', 'user.name=pphy96826-coder', '-c', 'user.email=pphy96826@gmail.com', 'commit', '-q', '-m',
        f'宗师: network generation {gen}\n\n{why}\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>')
    git('push', '-q', 'origin', 'main'); git('push', '-q', 'origin', 'main:gh-pages')
    log(f'gen {gen}: pushed {git("rev-parse", "--short", "HEAD")} to main and gh-pages')
    return True

def master_score(gen):
    """gen's score against the hand-written master (network only), from eval.py"""
    r = subprocess.run([sys.executable, os.path.join(HERE, 'eval.py'), f'gen{gen}.pt', '200', str(TEST_GAMES), 'master', '1500', '0'],
                       capture_output=True, text=True, cwd=ROOT)
    m = re.search(r'vs master/1500: (\d+)-(\d+)-(\d+)', r.stdout)
    if not m: raise RuntimeError('evaluation failed: ' + (r.stderr or r.stdout)[-300:])
    w, l, d = map(int, m.groups()); return (w + d / 2) / max(1, w + l + d), f'{w}-{l}-{d}'

def head_to_head(gen, online):
    """gen's score against the online generation, both network only"""
    r = subprocess.run([sys.executable, os.path.join(HERE, 'h2h.py'), f'gen{gen}.pt', f'gen{online}.pt', str(H2H_GAMES), '200'],
                       capture_output=True, text=True, cwd=ROOT)
    m = re.search(r'sims: (\d+)-(\d+)-(\d+)', r.stdout)
    if not m: raise RuntimeError('head-to-head failed: ' + (r.stderr or r.stdout)[-300:])
    w, l, d = map(int, m.groups()); return (w + d / 2) / max(1, w + l + d), f'{w}-{l}-{d}'

def watchdog():
    """restart training or the local play server if they stopped (unless training was stopped on purpose)"""
    alive = lambda pat: subprocess.run(['pgrep', '-f', pat], capture_output=True).returncode == 0
    if not alive('train/az/run_forever.sh') and not alive('train/az/train.py') and not os.path.exists(os.path.join(HERE, 'out', 'STOP')):
        subprocess.Popen([os.path.join(HERE, 'run_forever.sh')], cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        log('watchdog: training was not running, restarted it')
    if subprocess.run(['lsof', '-nP', '-iTCP:8765', '-sTCP:LISTEN'], capture_output=True).returncode != 0:
        subprocess.Popen([sys.executable, os.path.join(HERE, 'server.py'), '8765'], cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        log('watchdog: local play server was not running, restarted it')

def load_state():
    return json.load(open(STATE)) if os.path.exists(STATE) else {}
def save_state(st): json.dump(st, open(STATE, 'w'))

def main():
    st = load_state()
    if 'online_gen' not in st:
        # first run with the rule: the last version that went online before it (gen 36) is the reference; whatever
        # is online now must measure up to it, or the page goes back to it
        ref = 36; sc, txt = master_score(ref); st = {'online_gen': ref, 'online_score': sc, 'online_result': txt, 'tested': {}}
        log(f'reference: gen {ref} vs master {txt} ({sc:.0%})'); save_state(st)
        live = published_gen()
        if live != ref:
            sc2, txt2 = master_score(live); st['tested'][str(live)] = sc2
            if sc2 >= sc:
                st.update(online_gen=live, online_score=sc2, online_result=txt2); log(f'gen {live} (online) vs master {txt2} ({sc2:.0%}): keeps its place')
            else:
                log(f'gen {live} (online) vs master {txt2} ({sc2:.0%}) is worse than gen {ref}: rolling the page back')
                publish(ref, f'Rolled back: generation {live} scored {txt2} against the hand-written master, generation {ref} {txt}.')
            save_state(st)
    # republish the online version with the current page code (a no-op when nothing changed)
    publish(st['online_gen'], f'Page update for the online version (generation {st["online_gen"]}): players who beat 宗师 can send the game in as training data.')
    last_push = 0
    while True:
        try: watchdog()
        except Exception as e: log(f'watchdog failed: {e}')
        try:
            gen = json.load(open(os.path.join(WEB, 'net_best.json'))).get('gen')
            if gen is not None and gen != st['online_gen'] and str(gen) not in st['tested'] and time.time() - last_push >= MIN_GAP:
                pool = st.setdefault('history', [st['online_gen']])[-POOL:][::-1]   # current online version first
                res = {g: head_to_head(gen, g) for g in pool}; ms, mtxt = master_score(gen)
                avg = sum(v[0] for v in res.values()) / len(res); cur = res[pool[0]][0]
                txt = ', '.join(f'gen {g} {t} ({v:.0%})' for g, (v, t) in res.items())
                st['tested'][str(gen)] = {'pool': {str(g): v for g, (v, _) in res.items()}, 'master': ms}; save_state(st)
                if cur >= 0.5 and avg >= 0.5 and ms >= MASTER_FLOOR:
                    publish(gen, f'Promoted by training (beat the previous best >= 55% over 40 games). Against earlier online versions (random maps, randomised openings): {txt}; average {avg:.0%} (needs >= 50% against the current one and on average). Against the hand-written master: {mtxt} ({ms:.0%}, needs >= {MASTER_FLOOR:.0%}).')
                    st.update(online_gen=gen, online_score=ms, online_result=mtxt); st['history'].append(gen); save_state(st); last_push = time.time()
                else:
                    log(f'gen {gen}: {txt}, average {avg:.0%}; vs master {mtxt} ({ms:.0%}): not published')
        except Exception as e:
            log(f'publish failed: {e}')
        time.sleep(POLL)

if __name__ == '__main__':
    main()
