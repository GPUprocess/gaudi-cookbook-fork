import torch
import habana_frameworks.torch  # noqa
torch.ops.load_library("/data/glm53/q21_dist_LA/libmxk_op_lazy.so")
dev = torch.device("hpu")
# THE REFERENCE RULE (Q21-E30): compare against the bf16-QUANTIZED input,
# never the raw arange - bf16 (8-bit mantissa) quantizes arange(1000,1256)
# to runs of 3-5 identical values spaced 4 at magnitude 1024.
inp = torch.arange(1000, 1256, dtype=torch.float32).to(torch.bfloat16).unsqueeze(0).to(dev)
out = torch.ops.custom_op.store_test(inp)
o = out.cpu().float()
ref = torch.arange(1000, 1256, dtype=torch.float32).to(torch.bfloat16).float()[:128]
bad = []
for t in range(4):
    for k in range(8):
        if not torch.equal(o[t, k, 0:128], ref):
            bad.append((t, k, round((o[t, k, 0:128] - ref).abs().max().item(), 1)))
print("[ST] bad (t,k) blocks:", bad if bad else "NONE - all 32 bit-exact vs the bf16-quantized reference")
