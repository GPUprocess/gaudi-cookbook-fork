# Mandatory Session Protocol

## Purpose

This document defines the required start-of-session and end-of-session workflow for work on GLM-5.3-Flash optimization on Intel Gaudi 2.

It is intended to be copied into the root of the active project repository, or into its docs directory as specified by the repository's agent instructions.

## Mandatory reading before every work session

At the beginning of EVERY new agent session, before researching, editing, benchmarking, running commands that change system state, or proposing an implementation, the agent must:

1. Identify the active project repository and confirm its root directory.
2. Read the repository's AGENTS.md or equivalent agent instructions, if present.
3. Read the complete current TASKS.md.
4. Read the complete current EXPERIMENTS.md.
5. Read BASELINE.md and CURRENT_STATE.md, if present.
6. Read this protocol and the benchmark and hardware policies.
7. Inspect the current Git status, branch, remotes, and recent relevant commit history without changing them.
8. Summarize the current project state, the most relevant prior experiments, outstanding blockers, and the next proposed task before beginning implementation.

Do not rely on memory, previous chat context, or an earlier session's summary as a substitute for reading the current files.

If a required file is missing, unreadable, stale, or contradictory, report that fact and resolve it before proceeding. Do not silently create a replacement that discards existing project history.

## Locate the project and these policy files

The user may initially place these policy files in their home directory rather than the project repository. Search for the exact filenames in the user's home directory and the active workspace. Do not assume the current working directory is the repository root.

Once found:

- Preserve the user's originals.
- Copy the policy files into the root of the existing project repository, or into its docs directory where specified below.
- Do not create a new repository.
- Do not change Git remotes, branches, credentials, or the existing push workflow.
- If more than one plausible project repository or more than one copy of a policy file is found, stop and ask the user which one is authoritative.
- If the project repository cannot be identified confidently, report the paths inspected and ask the user for the repository location. Do not guess.
- After copying, verify that the repository copies match the home-directory originals, for example with checksums or a diff.

Recommended placement:

| File | Repository location |
|---|---|
| SESSION_PROTOCOL.md | docs/SESSION_PROTOCOL.md |
| BENCHMARK_POLICY.md | docs/BENCHMARK_POLICY.md |
| HARDWARE_ROADMAP.md | docs/HARDWARE_ROADMAP.md |
| BASELINE.md | BASELINE.md |

Update the repository's AGENTS.md, or existing master agent instruction file, with a short mandatory reference to these documents.

Do not overwrite or replace existing instructions.

If no AGENTS.md exists, propose creating one and preserve all existing project-specific instructions.

## Task selection and deduplication

After reading the required files:

1. Choose the next task based on measured bottlenecks, expected impact, risk, dependencies, and the backlog—not simply the lowest task number.
2. Search TASKS.md, EXPERIMENTS.md, Git history, existing experiment reports, and relevant code for prior work on the same underlying mechanism.
3. Check for equivalent experiments under different names. Do not repeat an experiment merely because its task ID or wording differs.
4. If a previous result is incomplete or inconclusive, decide whether a clearly justified controlled rerun is needed.
5. Update task status and experiment records as work progresses.

## One-variable-at-a-time discipline

For performance experiments, change one principal variable at a time wherever practical.

Record unavoidable confounders.

Preserve a known-good revision or a reversible patch before risky changes.

Separate exploratory experiments from accepted production implementation.

## End-of-session requirements

Before ending a work session:

- Update TASKS.md with status, evidence, blockers, and next action.
- Append a concise index entry to EXPERIMENTS.md for every experiment performed, including failed and inconclusive attempts.
- Store full experiment reports and raw benchmark outputs in the repository's established locations.
- Update BASELINE.md only when the baseline has been formally verified under the benchmark policy.
- Update CURRENT_STATE.md when implementation or deployment state materially changes.
- Record relevant architectural decisions in DECISIONS.md.
- Report changed files, commands and tests run, benchmark conditions, results, accuracy/correctness findings, unresolved issues, and Git status.
- Do not push until changes and outputs have been sanitized according to the project's security and privacy requirements.

## Safety and authorization

These procedures do not grant permission to disrupt production.

Do not reboot, reset HPUs, upgrade or change kernel/driver/firmware, terminate unrelated processes, alter production services, or interrupt existing workloads without explicit user approval.

Prefer read-only inspection and isolated testing.

Docker or a virtual environment does not isolate physical HPU faults or kernel/driver/firmware changes.
