# Autonomous Development and Environment Management Policy

## 1. Current Server Status

The entire 8x Intel Gaudi 2 server is dedicated to development,
research, experimentation, and optimization.

There is currently no production workload that requires protection.

The agent is authorized to use all 8 HPUs for development work,
including benchmarking, profiling, kernel development, MTP research,
model optimization, and alternative inference-engine testing.

The agent must not impose artificial HPU restrictions.

The 4-HPU configuration is a future optimization target, not a
restriction on current development.

## 2. Autonomous Engineering Authority

The agent is expected to independently:

1. Identify performance bottlenecks.
2. Research potential solutions.
3. Compare alternative implementations.
4. Design and prioritize experiments.
5. Implement improvements.
6. Run CPU and HPU tests.
7. Benchmark and analyze results.
8. Reject ineffective or regressive changes.
9. Document successful and unsuccessful experiments.
10. Update the roadmap and select subsequent tasks.

The agent must continue useful development work rather than waiting
for user approval for ordinary engineering decisions.

The agent may independently use all available HPUs when technically
justified.

## 3. Production Maintenance Restrictions Are Obsolete

Any previous requirement to wait for a production maintenance window
is obsolete.

The agent does not need production downtime approval to perform
development benchmarks, HPU measurements, experimental deployments,
or validation work.

In particular, it should proceed with previously blocked work when
the necessary preparation is complete, including:

- Stage-1 reproduction (A001).
- C015 HPU measurements.
- B023 HPU verification.
- I015 integration and boundary-capture testing.
- E015 A/B experiments.
- Paired accuracy evaluation.
- Remaining HPU-dependent optimization tasks.

The agent must review TASKS.md, CURRENT_STATE.md, and relevant
experiment records. It must remove obsolete maintenance-window
blockers from active task statuses and update the roadmap accordingly.

Do not erase historical records of why an experiment was previously
blocked. Preserve them as historical context.

## 4. Hardware and Experimentation

The agent may use 1, 2, 4, 6, or 8 HPUs as appropriate.

It should select configurations based on the experiment's objective
rather than defaulting to a fixed HPU count.

It may create isolated experimental environments, containers,
virtual environments, branches, and test services.

It must preserve known-good implementations and make experimental
changes reversible.

Before changing shared drivers, firmware, kernel components, or
system-wide dependencies, assess the potential impact and prefer
isolated alternatives where practical.

Do not perform destructive system changes without a clear technical
justification and recovery plan.

## 5. Clean Development Environment

Maintaining a clean and storage-efficient development environment
is a mandatory responsibility.

### Before downloading or building

- Check available disk space.
- Check whether required files already exist.
- Reuse model weights, repositories, build artifacts, and caches
  whenever practical.
- Avoid duplicate model downloads and unnecessary repository clones.
- Estimate storage requirements for large downloads and builds.
- Use clearly identified locations for temporary experiment data.

### During development

- Keep temporary files separate from persistent project files.
- Associate generated artifacts with their experiment.
- Avoid redundant model conversions and checkpoints.
- Avoid accumulating unnecessary build directories and logs.
- Do not leave abandoned containers or experimental services running.
- Monitor disk consumption during large operations.

### After experiments

Review and clean up resources created by the experiment.

Disposable resources may include:

- Temporary test files.
- Abandoned build directories.
- Intermediate model conversions that are no longer needed.
- Temporary downloads.
- Stopped experimental containers.
- Unused experimental Docker images and layers, where safe.
- Redundant temporary logs.
- Temporary virtual environments no longer required.

Never blindly run unrestricted Docker pruning or recursive deletion
against directories whose ownership or purpose is uncertain.

Never delete:

- Source code or uncommitted user changes.
- Experiment results needed for reproducibility.
- Verified model checkpoints.
- Model weights that are expensive or difficult to reacquire.
- Dependencies or caches used by active environments.
- Files whose ownership or purpose is uncertain.

When uncertain, preserve the artifact and document it.

## 6. Storage Management

Periodically inspect:

- Available filesystem capacity.
- Repository sizes.
- Model and checkpoint directories.
- Docker storage.
- Build caches.
- Experiment artifacts.
- Log directories.
- Temporary directories.

Maintain a lightweight inventory of major storage consumers.

Before a large download, build, or conversion, verify that sufficient
space is available.

Prioritize removing redundant temporary data over removing useful
research artifacts.

## 7. Git and Change Management

- Preserve the existing origin and upstream remotes.
- Do not push to upstream unless explicitly authorized.
- Use branches or otherwise reversible commits for experimental work.
- Do not overwrite unrelated user changes.
- Keep the working tree understandable.
- Commit coherent, tested improvements.
- Inspect diffs before committing or pushing.
- Sanitize credentials, private infrastructure details, and sensitive
  information before publishing changes.

## 8. Reproducibility

Every meaningful experiment must record:

- Unique experiment identifier.
- Objective and hypothesis.
- Source-code changes or commit reference.
- Configuration and environment.
- HPU count and resource requirements.
- Benchmark methodology and results.
- Accuracy or correctness results.
- Conclusion and next steps.
- Generated artifacts and cleanup performed.

Do not delete an experiment's only reproducible record.

Keep TASKS.md, EXPERIMENTS.md, CURRENT_STATE.md, BASELINE.md,
ACCURACY.md, and DECISIONS.md synchronized with actual development.

## 9. General Operating Principle

Operate as an autonomous performance-engineering team on a dedicated
research server.

Experiment broadly, use the available hardware, validate carefully,
and continuously improve the implementation.

The primary operational constraints are correctness, reproducibility,
code quality, and maintaining a clean development environment.

Aggressive experimentation must never become careless resource
management.
