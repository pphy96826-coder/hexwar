# First-move balance of maps: the best network plays itself (search 96 per move, some randomness in the first two
# rounds) and we count how often player 0 (青, first to deploy in round 1) wins.
#   python3 train/az/map_balance.py fixed <games per map>
#   python3 train/az/map_balance.py random <maps> <games per map>        (maps from train/az/maps.js)
import sys, os, json, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train as T
T.LOG = os.path.join(T.OUT, 'balance_log.txt')
kind = sys.argv[1]
if kind == 'fixed':
    maps, per = ['valley', 'plateau', 'canyon'], int(sys.argv[2])
elif kind == 'file':   # python3 train/az/map_balance.py file <maps.json> <games per map>
    maps, per = json.load(open(sys.argv[2])), int(sys.argv[3]); kind = os.path.splitext(os.path.basename(sys.argv[2]))[0]
else:
    n, per = int(sys.argv[2]), int(sys.argv[3])
    maps = json.loads(subprocess.check_output(['node', '-e', f"const {{randomMap}}=require('./maps');console.log(JSON.stringify(Array.from({{length:{n}}},()=>randomMap())))"], cwd=T.HERE))
_, res = T.run_node(dict(mode='selfplay', games=8, sims=96, noise=True, tempRounds=2, parallel=per, mapList=maps, perMap=per,
                         sides=[{'slot': 0}, {'slot': 0}]), {0: T.load('best.pt')})
out = {}
for r in res:
    d = out.setdefault(r['map'], {'p0': 0, 'p1': 0, 'draw': 0, 'rounds': 0})
    d['p0' if r['pw'] == 0 else 'p1' if r['pw'] == 1 else 'draw'] += 1; d['rounds'] += r['rounds']
defs = {m if isinstance(m, str) else m['id']: m for m in maps}
json.dump({'per': per, 'stats': out, 'maps': defs}, open(os.path.join(T.OUT, f'balance_{kind}.json'), 'w'))
for k, d in out.items():
    g = d['p0'] + d['p1'] + d['draw']
    print(f"{k:16s} games {g:3d}  青(先手) wins {d['p0']:3d} ({d['p0']/g:.0%})  赭 wins {d['p1']:3d}  draws {d['draw']}  avg rounds {d['rounds']/g:.1f}")
