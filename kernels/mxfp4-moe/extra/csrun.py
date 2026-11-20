import json
import torch
import habana_frameworks.torch  # noqa
torch.ops.load_library("/data/glm53/q21_dist_CS/libmxk_op_lazy.so")
dev = torch.device("hpu")
E, T, H, I2, K = 8, 4, 512, 512, 8
torch.manual_seed(20261114)
w13 = (torch.randn(E, I2, H) * 0.05).to(torch.float8_e4m3fn).to(dev)
s13 = (torch.rand(E, I2 // 128, H // 128) * 0.02 + 0.005).to(dev)
ids = torch.zeros(T, K, dtype=torch.int32).to(dev)  # all experts = 0
res = {}
# CS1: the constant store -> npts + the positions
o1 = torch.ops.custom_op.cs1(w13, s13, torch.zeros(T, H, dtype=torch.bfloat16).to(dev), ids).cpu().float()
pos = torch.nonzero((o1 - 3.0).abs() < 0.01).tolist()
res["cs1_n3.0_count"] = len(pos)
res["cs1_positions"] = pos[:40]
# CS2: the constant-acc reduce chain -> expect 128.0 at the same positions
o2 = torch.ops.custom_op.cs2(w13, s13, torch.zeros(T, H, dtype=torch.bfloat16).to(dev), ids).cpu().float()
p2 = torch.nonzero((o2 - 128.0).abs() < 0.5).tolist()
res["cs2_n128_count"] = len(p2)
res["cs2_positions_match_cs1"] = p2 == pos
# CS3: the FIXED compute - all-ones x, ids=0: raw[t,k,col] = sum_h w13[0,col,h]*1*s13[0,col//128,h//128]
xo = torch.ones(T, H, dtype=torch.bfloat16).to(dev)
o3 = torch.ops.custom_op.cs3(w13, s13, xo, ids).cpu().float()
wf = w13.float().cpu()
sf = s13.float().cpu()
s_exp = sf.repeat_interleave(128, 1).repeat_interleave(128, 2)[:, :I2, :H]
w_dq = (wf * s_exp).to(torch.bfloat16).float()
ref = w_dq[0].sum(dim=1)  # sum over H for expert 0
err = (o3[0, 0] - ref).abs().max().item()
res["cs3_ones_maxerr"] = err
res["cs3_ones_pass"] = bool(err <= 3e-2 * max(ref.abs().max().item(), 1e-6))
# CS3: the random-x full check (t=0..3, ids=0)
xr = (torch.randn(T, H) * 0.5).to(torch.bfloat16).to(dev)
o4 = torch.ops.custom_op.cs3(w13, s13, xr, ids).cpu().float()
xf = xr.float().cpu()
err4 = max(((o4[t, 0] - w_dq[0] @ xf[t]).abs().max().item()) for t in range(T))
res["cs3_randx_maxerr"] = err4
ref_max = (w_dq[0] @ xf[0]).abs().max().item()
res["cs3_randx_pass"] = bool(err4 <= 3e-2 * ref_max)
print(json.dumps(res, indent=1))
