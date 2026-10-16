# Evaluate a checkpoint against the hand-written AI without touching the running trainer.
#   python3 train/az/eval.py <ckpt> <sims> <games> [level] [budget]
import sys, os, json, time
sys.path.insert(0, os.path.dirname(__file__))
import train as T
ck, sims, games = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
level = sys.argv[4] if len(sys.argv) > 4 else 'master'; budget = int(sys.argv[5]) if len(sys.argv) > 5 else 1500
blend = float(sys.argv[6]) if len(sys.argv) > 6 else 0
threat = len(sys.argv) > 7 and sys.argv[7] == 'threat'
T.LOG = os.path.join(T.OUT, 'eval_log.txt'); T.SELFPLAY = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'selfplay.js')
net = T.load(ck); t = time.time()
_, res = T.run_node(dict(mode='arena', games=games, sides=[{'slot': 0, 'sims': sims, 'blend': blend, 'threat': threat}, {'hand': level, 'budget': budget}],
                         parallel=4, maps='mix', fixedShare=0.5), {0: net})
w = sum(r['winner'] == 0 for r in res); l = sum(r['winner'] == 1 for r in res)
T.log(f"{ck} @{sims} sims blend {blend} threat {threat} vs {level}/{budget}: {w}-{l}-{len(res)-w-l} in {time.time()-t:.0f}s")
for r in res: print(json.dumps(r))
