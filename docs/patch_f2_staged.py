#!/usr/bin/env python3
"""F2: eliminate the take_draft_token_ids RPC by piggybacking the draft ids on
ModelRunnerOutput (run on g2). Three files:

1. vllm/v1/outputs.py — the optional draft_token_ids field on ModelRunnerOutput.
2. vllm-gaudi hpu_model_runner.py — sample_tokens populates the field from the
   same self._draft_token_ids state that take_draft_token_ids() consumes, and
   take_draft_token_ids() keeps working (it clears after the read) so nothing
   else breaks.
3. vllm/v1/engine/core.py — post_step consumes model_output.draft_token_ids
   when present and only falls back to the RPC when it is absent.

Env-gated via GLM_F2_DRAFT_PIGGYBACK=1 (the default OFF keeps the old path).
"""
import ast

# ---------- 1. outputs.py ----------
p = "/opt/glm53/vllm/vllm/v1/outputs.py"
src = open(p).read()
old = """    # information related to cudagraph execution
    cudagraph_stats: CUDAGraphStat | None = None"""
new = """    # information related to cudagraph execution
    cudagraph_stats: CUDAGraphStat | None = None

    # [F2] draft token ids produced this step, piggybacked on the output so the
    # engine does not need the separate take_draft_token_ids RPC. None unless
    # the runner populated it (the GLM_F2_DRAFT_PIGGYBACK gate).
    draft_token_ids: "DraftTokenIds | None" = None"""
assert old in src, "outputs anchor"
src = src.replace(old, new, 1)
ast.parse(src)
open(p, "w").write(src)
print("1. outputs.py OK")

# ---------- 2. the runner ----------
p = "/opt/glm53/vllm-gaudi/vllm_gaudi/v1/worker/hpu_model_runner.py"
src = open(p).read()
# the sample_tokens return site: find the ModelRunnerOutput construction in the decode path
old = """        if self.is_driver_worker and self.profiler.enabled:"""
# we instead patch at the return of sample_tokens: find "return ModelRunnerOutput(" occurrences
import re
m = list(re.finditer(r"return ModelRunnerOutput\(", src))
print("   runner: ModelRunnerOutput constructions:", len(m))
# the safest: patch take_draft_token_ids to ALSO stash into a field the wrapper reads,
# and patch the sample_tokens tail where ModelRunnerOutput is built for the decode path.
# Locate the LAST construction (the decode-path return) and inject the piggyback.
idx = src.rfind("return ModelRunnerOutput(")
assert idx > 0, "no ModelRunnerOutput construction"
# insert before it: compute the draft ids if available
inject = """        # [F2] piggyback the draft ids on the output when gated (drops the
        # engine's separate take_draft_token_ids RPC).
        _f2_draft = None
        if os.environ.get("GLM_F2_DRAFT_PIGGYBACK", "0") == "1" and \\
                getattr(self, "_draft_token_ids", None) is not None:
            _dti = self._draft_token_ids
            _dti_list = _dti.tolist() if isinstance(_dti, torch.Tensor) else _dti
            _f2_draft = DraftTokenIds(list(self.input_batch.req_ids), _dti_list)
            self._draft_token_ids = None
"""
src = src[:idx] + inject + src[idx:]
# now add the field to the construction: find the first closing of this constructor call
tail = src[idx:]
tail_old = "return ModelRunnerOutput("
tail_new = "return ModelRunnerOutput(\n            draft_token_ids=_f2_draft,"
tail = tail.replace(tail_old, tail_new, 1)
src = src[:idx] + tail
ast.parse(src)
open(p, "w").write(src)
print("2. hpu_model_runner.py OK")

# ---------- 3. the engine ----------
p = "/opt/glm53/vllm/vllm/v1/engine/core.py"
src = open(p).read()
old = """    def post_step(self, model_executed: bool) -> None:
        # When using async scheduling we can't get draft token ids in advance,
        # so we update draft token ids in the worker process and don't
        # need to update draft token ids here.
        if self.check_for_draft_tokens and not self.async_scheduling and model_executed:
            draft_token_ids = self.model_executor.take_draft_token_ids()
            if draft_token_ids is not None:
                self.scheduler.update_draft_token_ids(draft_token_ids)"""
new = """    def post_step(self, model_executed: bool) -> None:
        # When using async scheduling we can't get draft token ids in advance,
        # so we update draft token ids in the worker process and don't
        # need to update draft token ids here.
        if self.check_for_draft_tokens and not self.async_scheduling and model_executed:
            # [F2] consume the piggybacked draft ids from the model output first
            # (drops the separate take_draft_token_ids RPC); fall back to the RPC
            # when the runner did not populate them.
            draft_token_ids = getattr(self, "_f2_last_draft", None)
            if draft_token_ids is None:
                draft_token_ids = self.model_executor.take_draft_token_ids()
            else:
                self._f2_last_draft = None
            if draft_token_ids is not None:
                self.scheduler.update_draft_token_ids(draft_token_ids)"""
assert old in src, "engine anchor"
src = src.replace(old, new, 1)
# stash the draft ids in step() right after model_output is obtained
old2 = """        # Before processing the model output, process any aborts that happened
        # during the model execution.
        self._process_aborts_queue()"""
new2 = """        # [F2] stash the piggybacked draft ids for post_step before the aborts.
        self._f2_last_draft = getattr(model_output, "draft_token_ids", None) \\
            if model_output is not None else None

        # Before processing the model output, process any aborts that happened
        # during the model execution.
        self._process_aborts_queue()"""
assert old2 in src, "engine step anchor"
src = src.replace(old2, new2, 1)
ast.parse(src)
open(p, "w").write(src)
print("3. core.py OK")
print("F2 patches applied")
