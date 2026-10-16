# Export a checkpoint for the browser: BatchNorm and input scaling folded into the convolutions, fp32, one blob.
#   python3 train/az/export_web.py <ckpt> <out.bin>     (+ writes <out.bin>.json with a few reference outputs)
import sys, os, json, struct, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import Net, fused, C_IN
ck, out = sys.argv[1], sys.argv[2]
net = Net(); net.load_state_dict(torch.load(ck, map_location='cpu')); net.eval()
f = fused(net, half=False)
norm = f.norm.view(-1)
ts = []
w = f.stem[0].weight.detach() / norm.view(1, -1, 1, 1); ts += [w, f.stem[0].bias.detach()]
for b in f.blocks: ts += [b.c1.weight, b.c1.bias, b.c2.weight, b.c2.bias]
ts += [f.pol[0].weight, f.pol[0].bias, f.pol[2].weight, f.pol[2].bias]
ts += [f.val[0].weight, f.val[0].bias, f.val[2].weight, f.val[2].bias]
ts += [f.end[0].weight, f.end[0].bias, f.end[2].weight, f.end[2].bias]
arr = np.concatenate([t.detach().float().numpy().ravel() for t in ts]).astype('<f4')
hdr = dict(blocks=len(f.blocks), f=f.stem[0].weight.shape[0], cin=C_IN, nt=f.pol[2].weight.shape[0], n=int(arr.size))
hb = json.dumps(hdr).encode()
with open(out, 'wb') as fo: fo.write(struct.pack('<I', len(hb))); fo.write(hb); fo.write(b'\0' * ((4 - (4 + len(hb)) % 4) % 4)); fo.write(arr.tobytes())
# reference outputs on a random valid-looking board for checking the JS implementations
torch.manual_seed(1); H, W = 11, 11
x = torch.randint(0, 3, (2, C_IN, H, W), dtype=torch.uint8); x[:, 0] = 1; x[:, 0, 0, ::2] = 0
with torch.no_grad(): v, e, lg = net(x)
json.dump(dict(H=H, W=W, x=x.numpy().ravel().tolist(), v=v.tolist(), e=e.tolist(), lg=lg.numpy().ravel().tolist()), open(out + '.json', 'w'))
print(hdr, os.path.getsize(out), 'bytes')
