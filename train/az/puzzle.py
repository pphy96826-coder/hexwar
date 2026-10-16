# python3 train/az/puzzle.py <ckpt> <sims> <blend> <threat 0/1>: the AI's turn in each puzzle position
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train as T
T.LOG = os.path.join(T.OUT, 'eval_log.txt')
ck, sims, blend, threat = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), sys.argv[4] == '1'
pz = json.load(open(os.path.join(T.OUT, 'puzzles.json')))
_, res = T.run_node(dict(mode='puzzle', games=8, puzzles=pz, sides=[{'slot': 0, 'sims': sims, 'blend': blend, 'threat': threat}]), {0: T.load(ck)})
f = lambda x: '∞' if x is None else str(x)
for r in sorted(res, key=lambda r: r['id']):
    print(f"{r['id']:24s} opp chain now {f(r['chainNow']):>3} / after resolve {f(r['chainNext']):>3}, opp bank {r['oppBank']:2d}, corr {r['corr']:6.1f} | {' '.join(r['placed'])}")
