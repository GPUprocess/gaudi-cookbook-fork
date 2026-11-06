# Hardware Development and Deployment Roadmap

## Objective

Find the best-performing, correct, stable implementation first using the available 8× Intel Gaudi 2 system, then optimize and validate a production deployment that uses 4 HPUs.

The 4-HPU deployment goal is a FINAL OPTIMIZATION PHASE, not an immediate development restriction.

## Development resource policy

- The agent may use up to all 8 HPUs for isolated development, implementation, profiling, scaling studies, and benchmarks when resources are available and the experiment is authorized.
- Do not impose an arbitrary 4-HPU ceiling during exploration.
- Use the smallest HPU allocation that answers a particular experiment efficiently, but expand to 6 or 8 HPUs when the hypothesis requires it.
- Record the exact HPU allocation for every run.
- Do not use HPUs currently occupied by production workloads unless the user has explicitly authorized that use.
- Do not disrupt the existing 1CatAI/vLLM-Gaudi DeepSeek V4.1 Flash service or any other production workload.
- If HPU allocation or isolation is uncertain, inspect process/device usage and ask before proceeding.

## Phased roadmap

### Phase 1 — Establish a trustworthy baseline

- Reconstruct and verify the current implementation and benchmark conditions.
- Reproduce the reported MTP=1 result against a controlled MTP-disabled configuration.
- Record performance, correctness, accuracy, resource use, and variability.
- Mark the reported approximately 21% batch-1 decode uplift as provisional until it passes the benchmark policy.

### Phase 2 — Find the best implementation

- Optimize the relevant bottlenecks without prematurely constraining HPU count.
- Explore 1, 2, 4, 6, and 8 HPU configurations where useful and safe.
- Investigate MTP depth only when supported by the actual model architecture/checkpoint.
- Separate decode latency, aggregate throughput, prefill, TTFT, and concurrency objectives.
- Profile before attempting low-level kernel or communication changes.
- Maintain correctness and accuracy gates throughout.

### Phase 3 — Select the best-performing candidate

- Compare candidates using verified, equivalent benchmark conditions.
- Select based on the intended deployment workload, not a single peak number.
- Document tradeoffs: latency, throughput, memory, context length, stability, utilization, and operational complexity.
- Preserve a reproducible known-good revision.

### Phase 4 — Optimize the selected implementation for 4 HPUs

Only after the best candidate is established:

- Reduce the HPU allocation to 4.
- Revisit model/expert placement, sharding, parallelism, communication, memory footprint, KV/cache/recurrent state, batching, and kernel choices.
- Measure the performance and quality cost of reducing from the best-performing configuration to 4 HPUs.
- Apply the same benchmark and correctness policy.
- Do not assume the 4-HPU version will retain 8-HPU performance.

### Phase 5 — Validate final deployment

The 4-HPU candidate is ready only when it has:

- Reproducible performance results on 4 HPUs.
- Accuracy and correctness within the agreed limits.
- Stable operation under representative workloads and context lengths.
- Documented memory headroom and failure behavior.
- Clear deployment, rollback, and monitoring instructions.
- No unapproved changes to host kernel, driver, firmware, or production services.

## Aspirational performance targets

The following are goals, not guaranteed outcomes or acceptance criteria by themselves:

- Approximately 200 tokens/sec decode.
- Approximately 3,000–6,000 tokens/sec prefill.
- Up to 1M context.

Record the workload, HPU count, batch/concurrency, and measurement definition beside any comparison to these targets.

Do not optimize one metric at the expense of correctness or the actual target workload without documenting the tradeoff.

## Production safety

The permission to use up to 8 HPUs for development does not authorize interruption of production.

Do not reboot, reset HPUs, upgrade or change kernel/driver/firmware, kill unrelated processes, or alter production services without explicit user approval.

Prefer isolated environments and reversible changes.

Containers and virtual environments do not isolate physical HPU faults or host-level driver/firmware changes.
