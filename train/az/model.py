"""Policy/value network for 蜂巢兵棋. Fully convolutional, so it works on any board size or terrain.
Input: uint8 planes [87, H, W] from game.js encode() (board turned so the player to move is at the bottom).
Outputs: value in [-1, 1] for the player to move; one logit per (unit type, cell); one logit for ending the turn."""
import torch, torch.nn as nn, torch.nn.functional as F

C_IN, NT = 87, 23
# divisors that bring the integer planes to ~[0, 1]
NORM = torch.ones(C_IN)
for ch, d in {57: 10, 58: 5, 59: 2, 63: 10, 73: 10, 74: 10, 75: 10, 76: 10, 77: 20, 78: 20, 79: 20, 82: 5, 83: 5, 86: 8}.items():
    NORM[ch] = d

class Block(nn.Module):
    def __init__(s, f):
        super().__init__()
        s.c1, s.b1 = nn.Conv2d(f, f, 3, padding=1, bias=False), nn.BatchNorm2d(f)
        s.c2, s.b2 = nn.Conv2d(f, f, 3, padding=1, bias=False), nn.BatchNorm2d(f)
    def forward(s, x, m):
        y = F.relu(s.b1(s.c1(x))) * m
        y = s.b2(s.c2(y)) * m
        return F.relu(x + y)

class Net(nn.Module):
    def __init__(s, f=64, blocks=6):
        super().__init__()
        s.register_buffer('norm', NORM.view(1, C_IN, 1, 1))
        s.stem = nn.Sequential(nn.Conv2d(C_IN, f, 3, padding=1, bias=False), nn.BatchNorm2d(f), nn.ReLU())
        s.blocks = nn.ModuleList([Block(f) for _ in range(blocks)])
        s.pol = nn.Sequential(nn.Conv2d(f, 32, 1), nn.ReLU(), nn.Conv2d(32, NT, 1))
        s.end = nn.Sequential(nn.Linear(2 * f, 64), nn.ReLU(), nn.Linear(64, 1))
        s.val = nn.Sequential(nn.Linear(2 * f, 128), nn.ReLU(), nn.Linear(128, 1))
    def forward(s, x):
        x = x.float() / s.norm
        m = x[:, 0:1]                                   # plane 0 = cell exists
        h = s.stem(x) * m
        for b in s.blocks:
            h = b(h, m)
        logits = s.pol(h)                               # [B, NT, H, W]
        cnt = m.sum((2, 3)).clamp(min=1)
        mean = h.sum((2, 3)) / cnt
        mx = (h - (1 - m) * 1e4).amax((2, 3))
        g = torch.cat([mean, mx], 1)
        return torch.tanh(s.val(g)).squeeze(1), s.end(g).squeeze(1), logits

def device():
    return torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

if __name__ == '__main__':
    import time
    dev = device(); net = Net().to(dev).eval()
    print('params', sum(p.numel() for p in net.parameters()))
    for B in (64, 256, 512):
        x = torch.randint(0, 2, (B, C_IN, 13, 15), dtype=torch.uint8).to(dev)
        with torch.no_grad():
            for _ in range(3): net(x)
            torch.mps.synchronize(); t = time.time(); n = 20
            for _ in range(n): v, e, l = net(x); l.float().cpu()
            torch.mps.synchronize()
        dt = (time.time() - t) / n
        print(f'batch {B}: {dt*1000:.1f} ms -> {B/dt:.0f} positions/s')

def fused(net, half=True):
    """Inference copy with BatchNorm folded into the convolutions (and fp16 weights): same outputs, fewer GPU kernels."""
    import copy
    from torch.nn.utils.fusion import fuse_conv_bn_eval
    n = copy.deepcopy(net).eval()
    for b in n.blocks:
        b.c1, b.b1 = fuse_conv_bn_eval(b.c1, b.b1), nn.Identity()
        b.c2, b.b2 = fuse_conv_bn_eval(b.c2, b.b2), nn.Identity()
    n.stem = nn.Sequential(fuse_conv_bn_eval(n.stem[0], n.stem[1]), nn.ReLU())
    if half:
        n = n.half(); n.norm.data = n.norm.data.float()
        n.forward = _half_forward.__get__(n)
    return n

def _half_forward(s, x):
    x = (x.float() / s.norm).half()
    m = x[:, 0:1]
    h = s.stem(x) * m
    for b in s.blocks:
        h = b(h, m)
    logits = s.pol(h)
    cnt = m.sum((2, 3)).clamp(min=1)
    mean = h.float().sum((2, 3)) / cnt.float()
    mx = (h - (1 - m) * 1e4).amax((2, 3)).float()
    g = torch.cat([mean, mx], 1).half()
    return torch.tanh(s.val(g)).squeeze(1).float(), s.end(g).squeeze(1).float(), logits.float()
