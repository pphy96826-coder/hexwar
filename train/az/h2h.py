# Head-to-head between two checkpoints (network only): python3 train/az/h2h.py <ckptA> <ckptB> <games> <sims>
# Built so a version cannot pass by learning its opponent's fixed lines: every few games a new random map, the first
# 4 decisions sampled from the search, and later a random pick among near-equal moves (visits within 10% of the best).
# Sides alternate. Prints A's result as W-L-D.
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train as T
T.LOG = os.path.join(T.OUT, 'eval_log.txt')
a, b, games, sims = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
_, res = T.run_node(dict(mode='arena', games=games, sides=[{'slot': 0, 'sims': sims}, {'slot': 1, 'sims': sims}], parallel=2,
                         maps='mix', fixedShare=0, arenaRandom=4, nearEqual=0.1), {0: T.load(a), 1: T.load(b)})
w = sum(r['winner'] == 0 for r in res); l = sum(r['winner'] == 1 for r in res); d = len(res) - w - l
T.log(f"{a} vs {b} @{sims} sims: {w}-{l}-{d} ({(w + d / 2) / max(1, len(res)):.0%}) on {len(set(r['map'] for r in res))} random maps")
