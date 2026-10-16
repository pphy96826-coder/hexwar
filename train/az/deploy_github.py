# Publishes each newly promoted network to GitHub (code on main, the playable page on gh-pages = GitHub Pages).
#   python3 train/az/deploy_github.py            (keeps running; checks every few minutes)
# A generation is promoted by train.py only after beating the previous best (>= 55% over 40 games); its export
# train/az/web/net_best.bin + net_best.json is what gets published. At most one push every MIN_GAP seconds.
import json, os, re, shutil, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
REPO = os.path.join(ROOT, 'hexwar-github'); WEB = os.path.join(HERE, 'web')
MIN_GAP = 3 * 3600; POLL = 300
LOG = os.path.join(HERE, 'out', 'deploy_log.txt')
CODE = {   # project file -> repository path (kept in sync on every publish)
    'hexwar_base.html': 'hexwar_base.html',
    **{f'train/az/{f}': f'train/az/{f}' for f in ['train.py', 'model.py', 'selfplay.js', 'game.js', 'mcts.js', 'maps.js', 'human_samples.js',
                                                 'eval.py', 'puzzle.py', 'export_web.py', 'server.py', 'run_forever.sh', 'deploy_github.py']},
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

def publish(gen):
    tmp = os.path.join(REPO, 'hexwar.html.tmp')
    subprocess.run([sys.executable, os.path.join(WEB, 'build.py'), os.path.join(ROOT, 'hexwar_base.html'), os.path.join(WEB, 'net_best.bin'), tmp],
                   check=True, capture_output=True)
    # the training engine must still load the page
    subprocess.run(['node', '-e', f"require({json.dumps(os.path.join(ROOT, 'train', 'engine'))})({json.dumps(tmp)})"], check=True, capture_output=True)
    os.replace(tmp, os.path.join(REPO, 'hexwar.html')); shutil.copyfile(os.path.join(REPO, 'hexwar.html'), os.path.join(REPO, 'index.html'))
    shutil.copyfile(os.path.join(WEB, 'net_best.bin'), os.path.join(REPO, 'models', 'net_best.bin'))
    shutil.copyfile(os.path.join(HERE, 'out', 'best.pt'), os.path.join(REPO, 'models', 'best.pt'))
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
        f'宗师: network generation {gen}\n\nPromoted by training after beating the previous best (>= 55% over 40 games).\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>')
    git('push', '-q', 'origin', 'main'); git('push', '-q', 'origin', 'main:gh-pages')
    log(f'gen {gen}: pushed {git("rev-parse", "--short", "HEAD")} to main and gh-pages')
    return True

def main():
    last_push = 0
    while True:
        try:
            info = json.load(open(os.path.join(WEB, 'net_best.json')))
            gen = info.get('gen')
            if gen is not None and gen != published_gen() and time.time() - last_push >= MIN_GAP:
                if publish(gen): last_push = time.time()
        except Exception as e:
            log(f'publish failed: {e}')
        time.sleep(POLL)

if __name__ == '__main__':
    main()
