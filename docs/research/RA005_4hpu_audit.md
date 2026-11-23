# RA-005 — The 4-HPU Target Physics Audit (Decision Document)

Status: **COMPLETE** (2026-11-23; the audit from the measured anchors + the corpus arithmetic)

## Question
Which of the owner's targets (200 t/s bs=1, 800 t/s @16) are reachable
on a 4×Gaudi2 node, and under which lever stacks?

## Measured anchors (the 4-HPU baseline)
- The Q14v2 boot (2026-11-09, TP4+EP4): **bs=1 41.7 t/s, @16 180.8 t/s**,
  the 100% suite — the measured 4-HPU baseline.
- ECF8 (the fused mode) on 4 HPU: the t/s delta small (bs=1 ~0-2%,
  @16 +2.5-4%); its real value is the **BG-7 1M-fit ENABLEMENT**
  (the weights 328.3 → ~282 GB; the 1M KV + the graphs fit with
  ~64 GB margin) — the capacity argument, not the throughput.

## The physics audit per target

### 200 t/s at bs=1 on 4 HPU
- The current 4-HPU bs=1: 41.7 t/s. The target: 4.8×.
- The step anatomy at bs=1 is host-paced (the replay floor 25-45 ms);
  the 4-card split REDUCES the per-card expert read (the routed-union
  physics) but the host floor is per-step, not per-card — the 4-card
  split does NOT shrink the host floor.
- The drafter gate (RA-049): AL ≥ 4.5-4.8 required at the lever-stack
  step times. The single MTP head cannot (the W7 ceiling [2.73, 3.3]).
- **VERDICT: NOT REACHABLE on 4 HPU** with the current drafter + the
  lever stack. The path: the self-trained drafter (RA-036/L6, the 3090
  hardware call) + the full RA-038-A node collapse + the D5-style
  routing. The 8-HPU node with the same stack has the same drafter
  gate — the card count is not the binding constraint at bs=1.

### 800 t/s at @16 on 4 HPU
- The current 4-HPU @16: 180.8 t/s. The target: 4.4×.
- The @16 step is device-paced; the 4-card split quadruples the
  per-card byte budget (the same 36-expert union per card at TP4+EP4:
  the routed union 7.2-10.5 GB/step/rank vs 3.6-5.2 at 8 cards).
- The node collapse (RA-052: the recipe 20.5-26.4k → 6-10k nodes)
  attacks the host replay floor (25-45 → 8-15 ms) — at @16 the host
  floor is hidden under the device span, so the node collapse helps
  bs=1 more than @16.
- The @16 device span: the MoE union read dominates; the 4-card split
  doubles the per-card traffic → the span grows. **The 8-HPU node is
  the @16 architecture; the 4-HPU node is structurally behind.**
- **VERDICT: NOT REACHABLE on 4 HPU** (the union-read physics); the
  8-HPU + the RA-052 node collapse + the drafter AL 4.5-4.8 is the
  only mapped route (the doc's 1315-1790 @16 at AL 4.5 on 4-HPU
  assumes the drafter + the byte savings the current stack lacks).

## The decision
1. The 4-HPU node: the capacity play (the 1M-context fit via ECF8),
   NOT the throughput play. Keep it for the 1M deployments.
2. The throughput targets: the 8-HPU node + the drafter decision
   (the RA-049 owner item) + the RA-052 node collapse.
3. The partial-offload variant: REJECTED (the PCIe wall: 12.4 GB/s
   per card = 1/197 of HBM; the 580-850 ms/step transfer at @16).
