# SPDX-License-Identifier: Apache-2.0
"""Block-FP8 dequant kernel runtime for GLM-5.3-Flash on vllm-gaudi (lazy mode).

Wired by install_block_fp8.py (the mxfp4 installer pattern):

* :func:`setup_env` at the top of ``vllm_gaudi/__init__.py`` — every vLLM process
  imports the platform plugin before the device opens, and the graph compiler
  reads ``GC_KERNEL_PATH`` at device open.
* :func:`install` at the end of ``vllm_gaudi/ops/hpu_fp8.py`` — loads the torch
  op library and verifies the op registered.
* :func:`_deferred_visibility_check` — a background thread that re-checks the
  op visibility 120 s after the install (the dispatcher init may be deferred;
  the Q21 diagnosis: the op was invisible at import but the worker context
  differs — EXP-Q21 probe addendum).
"""

import os
import threading
import time

ENABLED = os.environ.get("GLM_FP8_MXK", "0") == "1"
LIB_DIR = os.environ.get("GLM_FP8_MXK_LIB", "/tmp/mxk_dist")
TPC_LIB = "libmxk_tpc.so"
OP_LIB = "libmxk_op_lazy.so"
STOCK = "/usr/lib/habanalabs/libtpc_kernels.so"
TAG = "[glm-fp8-mxk]"
_installed = False
_deferred_done = False


def setup_env():
    """Register the kernel library with the graph compiler. Must run before the
    first HPU op of the process."""
    global ENABLED
    if not ENABLED:
        return
    tpc = os.path.join(LIB_DIR, TPC_LIB)
    if not os.path.isfile(tpc):
        ENABLED = False
        print(f"{TAG} GLM_FP8_MXK=1 but no {tpc}: staying on the stock ops")
        return
    cur = os.environ.get("GC_KERNEL_PATH", STOCK)
    rest = [p for p in cur.split(":") if p and p != tpc]
    if len(rest) == 1:
        os.environ["MXK_TPC_FWD_LIB"] = rest[0]
    elif len(rest) > 1:
        print(f"{TAG} GC_KERNEL_PATH had {len(rest)} libraries; forwarding to {STOCK} only")
    os.environ["GC_KERNEL_PATH"] = tpc
    rc = os.environ.get("PT_HPU_RECIPE_CACHE_CONFIG")
    if rc:
        path, _, rest_cfg = rc.partition(",")
        tag = "fp8mxk-"
        if path and tag not in path:
            path = path.rstrip("/") + "_" + tag + "/"
            try:
                os.makedirs(path, exist_ok=True)
                os.environ["PT_HPU_RECIPE_CACHE_CONFIG"] = path + ("," + rest_cfg if rest_cfg else "")
            except OSError as e:
                print(f"{TAG} cannot create recipe cache dir {path}: {e}; keeping {rc}")
    os.environ["PT_HPU_LAZY_ACC_PAR_MODE"] = "0"


def install():
    """Load the torch op library and verify the op registered. Called from the
    hook at the end of ops/hpu_fp8.py."""
    global _installed
    if not ENABLED or _installed:
        return
    _installed = True
    import torch

    op_lib = os.path.join(LIB_DIR, OP_LIB)
    if not os.path.isfile(op_lib):
        print(f"{TAG} op library missing ({op_lib}): stock ops kept")
        return
    try:
        torch.ops.load_library(op_lib)
    except Exception as e:
        print(f"{TAG} op library FAILED to load ({type(e).__name__}: {e}): stock ops kept")
        return
    co = torch.ops.custom_op
    ok = hasattr(co, "block_fp8_dequant_glm")
    print(f"{TAG} block_fp8_dequant_glm {'REGISTERED' if ok else 'NOT registered'} "
          f"from {op_lib}")
    # The deferred visibility check: the dispatcher init may be deferred; the
    # Q21 probe addendum showed the worker context differs from the probes.
    t = threading.Thread(target=_deferred_visibility_check, daemon=True)
    t.start()


def _deferred_visibility_check():
    """Re-check the op visibility after 120 s (the dispatcher may have fully
    initialized by then). Logs the result for the Q21 diagnosis."""
    global _deferred_done
    if _deferred_done:
        return
    _deferred_done = True
    time.sleep(120)
    import torch

    co = torch.ops.custom_op
    ok = hasattr(co, "block_fp8_dequant_glm")
    print(f"{TAG} DEFERRED CHECK (120s): block_fp8_dequant_glm "
          f"{'VISIBLE' if ok else 'still NOT visible'} in torch.ops.custom_op")
