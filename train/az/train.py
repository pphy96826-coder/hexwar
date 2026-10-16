"""AlphaZero-style training for 蜂巢兵棋.
    python3 train/az/train.py            # start or resume (checkpoints and log in train/az/out/)
Phases: 1) imitation of the hand-written AI to get started, 2) loop: self-play -> train -> new vs best arena
(promote at >= 55%), with periodic matches against the hand-written master to measure real strength."""
import os, sys, json, struct, subprocess, time, random, math, pickle
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(__file__))
from model import Net, device, C_IN, NT, fused

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, 'out'); os.makedirs(OUT, exist_ok=True)
SELFPLAY = os.path.join(HERE, 'selfplay.js'); LOG = os.path.join(OUT, 'log.txt'); STATE = os.path.join(OUT, 'state.json')
DEV = device(); END = 65535
CFG = dict(imit_steps=3000, max_gen=10**9, imit_games=640, imit_budget=300, sp_games=384, sims=160, arena_games=40, arena_sims=96, train_steps=1200,
           batch=256, lr=1e-3, buffer=220000, promote=0.55, master_every=3, master_games=48,
           league_games=96, league_budget=500, revert=0.35,
           striker_games=64, zmix=0.5, human_per_batch=12)

def log(m):
    line = time.strftime('[%m-%d %H:%M:%S] ') + m
    print(line, flush=True); open(LOG, 'a').write(line + '\n')

def read_exact(f, n):
    b = bytearray()
    while len(b) < n:
        c = f.read(n - len(b))
        if not c: raise EOFError('node process ended')
        b += c
    return bytes(b)

@torch.no_grad()
def infer(pl, models):
    n = struct.unpack_from('<I', pl, 0)[0]; o = 4; reqs = []
    for _ in range(n):
        slot, H, W = pl[o], pl[o + 1], pl[o + 2]; L = struct.unpack_from('<H', pl, o + 3)[0]; o += 5
        x = np.frombuffer(pl, np.uint8, C_IN * H * W, o).reshape(C_IN, H, W); o += C_IN * H * W
        idx = np.frombuffer(pl, np.uint16, L, o).astype(np.int64); o += 2 * L
        reqs.append((slot, H, W, x, idx))
    out = [None] * n
    for slot in set(r[0] for r in reqs):
        ids = [i for i, r in enumerate(reqs) if r[0] == slot]
        Hm = max(reqs[i][1] for i in ids); Wm = max(reqs[i][2] for i in ids)
        X = np.zeros((len(ids), C_IN, Hm, Wm), np.uint8)
        for j, i in enumerate(ids): _, H, W, x, _ = reqs[i]; X[j, :, :H, :W] = x
        v, e, lg = models[slot](torch.from_numpy(X).to(DEV))
        v, e, lg = v.float().cpu().numpy(), e.float().cpu().numpy(), lg.float().reshape(len(ids), -1).cpu().numpy()
        for j, i in enumerate(ids):
            _, H, W, _, idx = reqs[i]
            isend = idx == END; t = idx // (H * W); rem = idx % (H * W); y = rem // W; xx = rem % W
            flat = np.where(isend, 0, t * Hm * Wm + y * Wm + xx)
            vals = np.where(isend, e[j], lg[j][flat]).astype(np.float32)
            out[i] = struct.pack('<f', float(v[j])) + vals.tobytes()
    return b''.join(out)

def parse_samples(pl):
    hl = struct.unpack_from('<I', pl, 0)[0]; meta = json.loads(pl[4:4 + hl]); o = 4 + hl; res = []
    for m in meta:
        H, W = m['H'], m['W']; sz = C_IN * H * W
        x = np.frombuffer(pl, np.uint8, sz, o).reshape(C_IN, H, W).copy(); o += sz
        z = float(m['z']) if m.get('q') is None else CFG['zmix'] * float(m['z']) + (1 - CFG['zmix']) * float(m['q'])
        res.append((x, np.array(m['idx'], np.int64), np.array(m['pi'], np.float32), z, float(m.get('pw', 1))))
    return res

def run_node(cfg, models, tries=3):
    for k in range(tries):
        try: return run_node1(cfg, models)
        except (EOFError, BrokenPipeError, struct.error) as e:
            log(f"  game process failed ({e!r}), retrying ({k+1}/{tries})"); time.sleep(5)
    raise RuntimeError('game process kept failing')

def run_node1(cfg, models):
    models = {k: fused(v) for k, v in models.items()}   # BatchNorm folded, fp16: ~2x faster inference
    p = subprocess.Popen(['node', SELFPLAY, json.dumps(cfg)], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    samples, results = [], []
    try:
        while True:
            h = read_exact(p.stdout, 5); typ = h[:1]; ln = struct.unpack('<I', h[1:])[0]; pl = read_exact(p.stdout, ln)
            if typ == b'Q':
                a = infer(pl, models); p.stdin.write(b'A' + struct.pack('<I', len(a)) + a); p.stdin.flush()
            elif typ == b'D': samples += parse_samples(pl)
            elif typ == b'R': results = json.loads(pl)
            elif typ == b'X': break
    finally:
        p.wait()
    return samples, results

def batch_tensors(batch):
    B = len(batch); Hm = max(b[0].shape[1] for b in batch); Wm = max(b[0].shape[2] for b in batch); Lm = max(len(b[1]) for b in batch)
    X = np.zeros((B, C_IN, Hm, Wm), np.uint8); IDX = np.zeros((B, Lm), np.int64); PI = np.zeros((B, Lm), np.float32)
    MASK = np.zeros((B, Lm), bool); Z = np.zeros(B, np.float32); PW = np.ones(B, np.float32); NTOT = NT * Hm * Wm
    for j, smp in enumerate(batch):
        x, idx, pi, z = smp[:4]; PW[j] = smp[4] if len(smp) > 4 else 1.0
        _, H, W = x.shape; X[j, :, :H, :W] = x
        isend = idx == END; t = idx // (H * W); rem = idx % (H * W)
        flat = np.where(isend, NTOT, t * Hm * Wm + (rem // W) * Wm + rem % W)
        IDX[j, :len(idx)] = flat; PI[j, :len(idx)] = pi; MASK[j, :len(idx)] = True; Z[j] = z
    t = lambda a: torch.from_numpy(a).to(DEV)
    return t(X), t(IDX), t(PI), t(MASK), t(Z), t(PW)

def train(net, buf, steps):
    net.train(); opt = torch.optim.AdamW(net.parameters(), lr=CFG['lr'], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    vl = pl = 0.0
    for s in range(steps):
        nh = min(CFG['human_per_batch'], len(HUMAN))   # the player's own games are in every batch
        batch = [buf[i] for i in np.random.randint(0, len(buf), CFG['batch'] - nh)] + [HUMAN[i] for i in np.random.randint(0, max(1, len(HUMAN)), nh)]
        X, IDX, PI, MASK, Z, PW = batch_tensors(batch)
        v, e, lg = net(X)
        flat = torch.cat([lg.reshape(lg.shape[0], -1), e.unsqueeze(1)], 1)
        g = torch.gather(flat, 1, IDX).masked_fill(~MASK, -1e9)
        ploss = (-(PI * F.log_softmax(g, 1)).sum(1) * PW).sum() / PW.sum().clamp(min=1)
        vloss = F.mse_loss(v, Z)
        opt.zero_grad(); (ploss + vloss).backward(); opt.step(); sched.step()
        vl += vloss.item(); pl += ploss.item()
    net.eval(); return vl / steps, pl / steps

def arena(a, b, games, sims, mode_b=None):
    """a vs b (b is a network, or {'hand': level} for the hand-written AI). Returns a's score in [0,1]."""
    sides = [{'slot': 0, 'sims': sims}, mode_b if isinstance(mode_b, dict) else {'slot': 1, 'sims': sims}]
    _, res = run_node(dict(mode='arena', games=games, sides=sides, parallel=4, maps='mix', fixedShare=0.5), {0: a, 1: b} if b else {0: a})
    w = sum(1 for r in res if r['winner'] == 0); l = sum(1 for r in res if r['winner'] == 1); d = len(res) - w - l
    return (w + d / 2) / max(1, len(res)), w, l, d

def publish_web(gen):
    # the network the web page loads (served by server.py at /api/net); written atomically
    web = os.path.join(HERE, 'web'); tmp = os.path.join(web, 'net_best.tmp')
    r = subprocess.run([sys.executable, os.path.join(HERE, 'export_web.py'), os.path.join(OUT, 'best.pt'), tmp], capture_output=True, text=True)
    if r.returncode == 0:
        os.replace(tmp, os.path.join(web, 'net_best.bin'))
        if os.path.exists(tmp + '.json'): os.remove(tmp + '.json')
        json.dump({'gen': gen, 'time': time.strftime('%m-%d %H:%M')}, open(os.path.join(web, 'net_best.json'), 'w'))
        log(f"  web page network updated to gen {gen}")
    else: log(f"  web export failed: {r.stderr[-300:]}")

def save(net, name): torch.save(net.state_dict(), os.path.join(OUT, name))
def load(name):
    n = Net().to(DEV); n.load_state_dict(torch.load(os.path.join(OUT, name), map_location=DEV)); n.eval(); return n

HUMAN = []; HUMAN_T = 0
def load_human():
    global HUMAN, HUMAN_T
    HUMAN_T = time.time()
    hb = os.path.join(OUT, 'human.bin')
    r = subprocess.run(['node', os.path.join(HERE, 'human_samples.js'), hb], capture_output=True, text=True)
    if os.path.exists(hb):
        z0 = CFG['zmix']; HUMAN = parse_samples(open(hb, 'rb').read())
    log(f"player games: {len(HUMAN)} positions ({r.stderr.strip().splitlines()[-1] if r.stderr.strip() else ''})")

def main():
    load_human()
    if not os.path.exists(os.path.join(HERE, 'web', 'net_best.bin')): publish_web(json.load(open(STATE)).get('best', 21) if os.path.exists(STATE) else 0)
    state = json.load(open(STATE)) if os.path.exists(STATE) else {'gen': 0}
    buf = []
    if not os.path.exists(os.path.join(OUT, 'best.pt')):
        log(f"imitation: {CFG['imit_games']} games of the hand-written AI (normal)")
        t = time.time()
        samples, res = run_node(dict(mode='imitate', games=CFG['imit_games'], parallel=2, maps='mix',
                                     sides=[{'hand': 'normal', 'budget': CFG['imit_budget']}] * 2), {})
        log(f"  {len(samples)} positions in {time.time()-t:.0f}s")
        buf = samples
        net = Net().to(DEV); vl, pl = train(net, buf, CFG['imit_steps'])
        log(f"  trained: value loss {vl:.3f}, policy loss {pl:.3f}")
        save(net, 'best.pt'); save(net, 'gen0.pt')
        sc, w, l, d = arena(net, None, CFG['master_games'], CFG['arena_sims'], {'hand': 'hard', 'budget': 800})
        log(f"  imitation net vs hand-written hard: {w}-{l}-{d} ({sc*100:.0f}%)")
    best = load('best.pt'); cur = load('cur.pt') if os.path.exists(os.path.join(OUT, 'cur.pt')) else load('best.pt')
    BUF = os.path.join(OUT, 'buffer.pkl')
    if not buf and os.path.exists(BUF):
        buf = pickle.load(open(BUF, 'rb')); log(f"resumed with {len(buf)} buffered positions")
    while True:
        g = state['gen'] + 1; t = time.time()
        if g > state['gen'] + 1 or os.path.getmtime(os.path.join(HERE, 'games')) > HUMAN_T: load_human()   # new game records
        ov = os.path.join(OUT, 'cfg.json')   # settings can be changed between generations without a restart
        if os.path.exists(ov):
            new = json.load(open(ov)); changed = {k: v for k, v in new.items() if CFG.get(k) != v}
            if changed: CFG.update(changed); log(f"settings changed: {changed}")
        games = CFG['sp_games'] * (2 if len(buf) < CFG['buffer'] // 4 else 1)   # after a restart the buffer is empty
        samples, res = run_node(dict(mode='selfplay', games=games, sims=CFG['sims'], noise=True, tempRounds=3,
                                     parallel=48, maps='mix', sides=[{'slot': 0}, {'slot': 0}]), {0: best})
        rounds = sum(r['rounds'] for r in res) / max(1, len(res))
        log(f"gen {g}: self-play {len(res)} games ({len(samples)} positions, avg {rounds:.1f} rounds) in {time.time()-t:.0f}s")
        if CFG['league_games']:
            # games against the hand-written master (only the network's moves become training positions), so the
            # network keeps learning to punish and to survive its rushes and kill combinations
            t = time.time()
            ls, lr = run_node(dict(mode='selfplay', games=CFG['league_games'], sims=CFG['sims'], noise=True, tempRounds=3,
                                   parallel=12, maps='mix', sides=[{'slot': 0}, {'hand': 'master', 'budget': CFG['league_budget']}]), {0: best})
            w = sum(r['winner'] == 0 for r in lr)
            log(f"  league vs hand master: {w}/{len(lr)} won during training games ({len(ls)} positions) in {time.time()-t:.0f}s")
            samples += ls
        if CFG['striker_games']:
            # a scripted all-in attacker (rams behind the base, catapults, barrels, dragon): learn to stop such rushes
            t = time.time()
            ss, sr = run_node(dict(mode='selfplay', games=CFG['striker_games'], sims=CFG['sims'], noise=True, tempRounds=3,
                                   parallel=16, maps='mix', sides=[{'slot': 0}, {'hand': 'striker'}]), {0: best})
            w = sum(r['winner'] == 0 for r in sr)
            log(f"  vs scripted attacker: {w}/{len(sr)} won ({len(ss)} positions) in {time.time()-t:.0f}s")
            samples += ss
        buf += samples; buf = buf[-CFG['buffer']:]
        t = time.time(); vl, pl = train(cur, buf, CFG['train_steps'])
        log(f"  trained {CFG['train_steps']} steps in {time.time()-t:.0f}s: value loss {vl:.3f}, policy loss {pl:.3f}")
        sc, w, l, d = arena(cur, best, CFG['arena_games'], CFG['arena_sims'])
        promoted = sc >= CFG['promote']
        log(f"  new vs best: {w}-{l}-{d} ({sc*100:.0f}%) -> {'PROMOTED' if promoted else 'kept old best'}")
        if promoted:
            best.load_state_dict(cur.state_dict()); save(best, 'best.pt'); save(best, f'gen{g}.pt'); state['best'] = g; publish_web(g)
        elif sc < CFG['revert']:
            cur.load_state_dict(best.state_dict())   # clearly worse: start again from the best
        # otherwise keep training the candidate, so small gains add up over generations
        save(cur, 'cur.pt')
        state['gen'] = g; json.dump(state, open(STATE, 'w'))
        if g % 2 == 0:   # so a crash loses at most two generations of games
            pickle.dump(buf, open(BUF + '.tmp', 'wb'), protocol=4); os.replace(BUF + '.tmp', BUF)
        if g % CFG['master_every'] == 0:
            sc, w, l, d = arena(best, None, CFG['master_games'], 200, {'hand': 'master', 'budget': 1500})
            log(f"  best vs hand-written master: {w}-{l}-{d} ({sc*100:.0f}%)")
        if os.path.exists(os.path.join(OUT, 'STOP')):
            pickle.dump(buf, open(BUF + '.tmp', 'wb'), protocol=4); os.replace(BUF + '.tmp', BUF); log(f'stop file found, buffer saved ({len(buf)} positions), exiting'); break
        if g >= CFG['max_gen']: break

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'smoke':
        CFG.update(imit_games=16, sp_games=16, arena_games=8, master_games=4, train_steps=50, master_every=1, imit_steps=50, max_gen=1, sims=32, arena_sims=32)
        OUT = os.path.join(HERE, 'out_smoke'); os.makedirs(OUT, exist_ok=True); LOG = os.path.join(OUT, 'log.txt'); STATE = os.path.join(OUT, 'state.json')
    main()
