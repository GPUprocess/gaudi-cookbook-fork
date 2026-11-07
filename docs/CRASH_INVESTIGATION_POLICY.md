# Crash Investigation, Diagnosis and Autonomous Recovery Policy

## 1. Purpose

The agent must behave as a self-directed systems engineer when encountering runtime failures.

A crash is an engineering investigation, not an instruction to restart the service and wait.

The objective is to identify the underlying cause, establish a reproducible explanation, implement a correction where possible, and verify that the system is healthy.

The development server is entirely dedicated to research and development. There is no production maintenance window or production availability requirement.

## 2. Mandatory GaudiGrimoire MCP Consultation

GaudiGrimoire MCP is a primary technical research resource and must be consulted during Gaudi-related troubleshooting.

For every significant Gaudi, Habana runtime, HPU, kernel, graph, memory, synchronization, or vLLM-Gaudi failure:

1. Search for the exact error message or error code.
2. Search for the affected subsystem.
3. Search for related known issues and workarounds.
4. Investigate relevant Gaudi 2 architectural limitations.
5. Compare recommendations with the installed software versions.
6. Inspect referenced implementations when available.
7. Record relevant findings and their source references.

Do not assume that a suggested workaround applies to the installed software version.

If GaudiGrimoire has no relevant information, continue with upstream source code, official documentation, GitHub issues, and other technical references.

Never invent MCP findings or claim a search was performed when it was not.

## 3. Crash Investigation Procedure

Whenever an important process crashes, hangs, produces unexpected device errors, or fails to initialize, follow this procedure.

### Phase A: Preserve evidence

Before restarting the engine, capture:

- Complete engine logs around the failure.
- Python tracebacks and native stack traces.
- HPU health and allocation information.
- `hl-smi` output.
- Relevant `dmesg` and journal entries.
- Process status and resource consumption.
- Runtime, driver, firmware, and framework versions.
- Model and engine configuration.
- Environment variables relevant to execution.
- Recent source-code changes and Git commit.
- The exact command that triggered the failure.

Preserve the original evidence before performing cleanup.

Sanitize sensitive information before publishing it.

### Phase B: Form hypotheses

Do not jump directly to restarting or reverting.

Generate at least three plausible hypotheses when the available evidence supports multiple explanations.

Examples:

- Invalid HPU graph capture or replay.
- Incorrect tensor shape or memory layout.
- HBM exhaustion or fragmentation.
- Incorrect stream synchronization.
- Unsupported kernel operation.
- Runtime/driver incompatibility.
- Invalid recurrent-state or KV-cache handling.
- Incorrect resource allocation.
- A regression introduced by a recent code change.

For each hypothesis, record:

- Supporting evidence.
- Contradicting evidence.
- Confidence level.
- How to test it.
- Expected observations if correct.
- Expected observations if incorrect.

Do not force three artificial hypotheses when evidence clearly identifies a single cause.

### Phase C: Investigate

Consult GaudiGrimoire and inspect the actual implementation.

Compare the failing execution path with the last known-good implementation.

Use Git history and experiment records to identify the changes associated with the failure.

Determine whether the failure is:

- Software-related.
- Hardware-related.
- Configuration-related.
- Resource-related.
- An interaction between multiple components.
- Still undetermined.

### Phase D: Controlled reproduction

Create the smallest reproduction that can demonstrate the failure.

Change one meaningful variable at a time.

Prefer isolated experiments and reversible Git branches.

Do not repeat a crash-inducing experiment indefinitely without extracting additional diagnostic information.

### Phase E: Recovery and verification

Restart the experimental engine when appropriate.

After restarting:

1. Verify that all expected HPUs are visible.
2. Verify that the model loads successfully.
3. Perform a representative inference request.
4. Check for new errors.
5. Verify that the process remains healthy.
6. Confirm that the original failure condition has been addressed.

A successful restart alone does not establish root cause.

If the original failure cannot be reproduced, mark the investigation INCONCLUSIVE rather than VERIFIED.

## 4. Restart Policy

Restarting vLLM is an operational recovery action, not a diagnostic conclusion.

Never repeatedly restart the engine without investigating why it failed.

If a restart does not resolve the issue:

- Preserve the new failure evidence.
- Compare it with the original failure.
- Update the hypotheses.
- Investigate alternative explanations.
- Avoid repeating identical restart attempts.

Do not wait indefinitely for the engine to become healthy.

Continue independent research, code inspection, documentation, benchmark preparation, and other eligible development work.

## 5. Root Cause Classification

Every significant incident must receive one of these classifications:

- CONFIRMED ROOT CAUSE
- PROBABLE ROOT CAUSE
- MULTIPLE CONTRIBUTING FACTORS
- UNRESOLVED
- TRANSIENT / NOT REPRODUCIBLE

Clearly distinguish observations from assumptions.

A root cause is confirmed only when supported by sufficient evidence, preferably through controlled reproduction or a verified corrective change.

## 6. Documentation

Every significant incident receives a unique incident or experiment ID.

Record:

- Incident identifier.
- Date.
- Affected component.
- Last known-good commit.
- Failing commit.
- Exact error.
- HPU allocation.
- Relevant environment versions.
- GaudiGrimoire searches and findings.
- Initial hypotheses.
- Experiments performed.
- Recovery actions.
- Root cause classification.
- Corrective changes.
- Verification results.
- Remaining risks.
- Follow-up tasks.

Update TASKS.md, EXPERIMENTS.md, CURRENT_STATE.md, and relevant detailed reports.

Preserve failed attempts as searchable engineering knowledge.

## 7. Autonomous Continuation

A runtime failure must not automatically stop the entire optimization project.

When an engine failure prevents one experiment:

1. Determine exactly which tasks are affected.
2. Mark only those tasks as BLOCKED.
3. Identify independent tasks that remain executable.
4. Continue those tasks.
5. Investigate the failure in parallel where practical.
6. Reassess the task queue after each meaningful finding.

Never mark unrelated completed benchmarks as blocked by a subsequent engine failure.

Never rerun a previously completed benchmark merely because the current engine is unhealthy.

## 8. Final Principle

The agent must diagnose before concluding, form hypotheses before making arbitrary changes, consult GaudiGrimoire before inventing Gaudi-specific solutions, and continue useful work whenever an individual experiment becomes blocked.

Restarting a service is not a substitute for engineering investigation.
