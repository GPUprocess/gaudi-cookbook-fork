# SPDX-License-Identifier: Apache-2.0
"""Block-FP8 dequant kernel runtime for GLM-5.3-Flash on vllm-gaudi (lazy mode).

Wired by install_block_fp8.py (the mxfp4 installer pattern):

* :func:`setup_env` at the top of ``vllm_gaudi/__init__.py`` — every vLLM process
  imports the platform plugin before the device opens, and the graph compiler
  reads ``GC_KERNEL_PATH`` at device open.
* :func:`install` at the end of ``vllm_gaudi/ops/hpu_fp8.py`` — loads the torch
  op library and verifies the op registered.

Switch (off by default; off means nothing loads and the stock ops run):

``GLM_FP8_MXK=1``
    Load ``libmxk_tpc.so`` (carrying ``block_fp8_dequant_glm`` among the gated
    GUIDs) via ``GC_KERNEL_PATH`` and ``libmxk_op_lazy.so`` via
    ``torch.ops.load_library``. ``GLM_FP8_MXK_LIB`` overrides the build
    directory (default /tmp/mxk_dist).
"""

import os

ENABLED = os.environ.get("GLM_FP8_MXK", "0") == "1"
LIB_DIR = os.environ.get("GLM_FP8_MXK_LIB", "/tmp/mxk_dist")
TPC_LIB = "libmxk_tpc.so"
OP_LIB = "libmxk_op_lazy.so"
STOCK = "/usr/lib/habanalabs/libtpc_kernels.so"
TAG = "[glm-fp8-mxk]"
_installed = False


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
    # One file in GC_KERNEL_PATH (required with the recipe cache on); the
    # library forwards stock GUIDs itself.
    cur = os.environ.get("GC_KERNEL_PATH", STOCK)
    rest = [p for p in cur.split(":") if p and p != tpc]
    if len(rest) == 1:
        os.environ["MXK_TPC_FWD_LIB"] = rest[0]
    elif len(rest) > 1:
        print(f"{TAG} GC_KERNEL_PATH had {len(rest)} libraries; forwarding to {STOCK} only")
    os.environ["GC_KERNEL_PATH"] = tpc
    # Recipe cache keys do not cover the kernel library: one cache dir per build.
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
    # Required: with the lazy accumulation thread on, a custom op whose input is
    # a view of a tensor produced in the same lazy graph fails intermittently.
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
