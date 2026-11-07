#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Install the block-FP8 dequant kernel hooks into the deployed vllm_gaudi.

    python3 install_block_fp8.py [--check]

The mxfp4 installer pattern adapted to our op (see the ANCHORS docstring).
"""
import argparse
import importlib.util
import os
import sys

MARK = "[glm-fp8-mxk]"
RT_NAME = "block_fp8_rt.py"

INIT_HOOK = (
    f"# {MARK} register the block-FP8 kernel library before the first HPU op (inert unless GLM_FP8_MXK=1)\n"
    "try:\n"
    "    from vllm_gaudi.ops.block_fp8_rt import setup_env as _fp8mxk_setup_env\n"
    "    _fp8mxk_setup_env()\n"
    "except Exception as _fp8mxk_e:  # never break the plugin import\n"
    f"    print(f'{MARK} setup_env failed: {{_fp8mxk_e}}')\n"
)

OPS_HOOK = (
    f"\n# {MARK} decode-sized block-FP8 calls -> custom TPC kernels (inert unless GLM_FP8_MXK=1)\n"
    "from vllm_gaudi.ops import block_fp8_rt as _fp8mxk_rt\n"
    "_fp8mxk_rt.install()\n"
)

ANCHORS = [
    ("ops/hpu_fp8.py", "[gaudi-fix:fp8-moe-gather]"),
    ("ops/hpu_fp8.py", "_HPU_MOE_GATHER"),
    ("__init__.py", "def register"),
]


def pkg_dir(name):
    spec = importlib.util.find_spec(name)
    return list(spec.submodule_search_locations)[0] if spec and spec.submodule_search_locations else None


def purge_pyc(path):
    pyc = os.path.join(os.path.dirname(path), "__pycache__")
    base = os.path.splitext(os.path.basename(path))[0]
    try:
        for fn in os.listdir(pyc):
            if fn.startswith(base + "."):
                os.remove(os.path.join(pyc, fn))
    except OSError:
        pass


def read(path):
    with open(path) as f:
        return f.read()


def write(path, text):
    with open(path, "w") as f:
        f.write(text)
    purge_pyc(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    root = pkg_dir("vllm_gaudi")
    if not root:
        sys.exit(f"{MARK} vllm_gaudi not found; refusing")
    rt_src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), RT_NAME)
    if not os.path.isfile(rt_src_path):
        sys.exit(f"{MARK} {rt_src_path} not found; refusing")
    rt_src = read(rt_src_path)
    compile(rt_src, rt_src_path, "exec", dont_inherit=True)

    missing = []
    for rel, need in ANCHORS:
        p = os.path.join(root, rel)
        if not os.path.isfile(p) or need not in read(p):
            missing.append(f"{rel}: {need!r}")
    if missing:
        print(f"{MARK} vllm_gaudi at {root} does not match; refusing to patch:", file=sys.stderr)
        for m in missing:
            print(f"  missing {m}", file=sys.stderr)
        sys.exit(1)
    if args.check:
        print(f"{MARK} all {len(ANCHORS)} anchors found in {root}")
        return

    ops_py = os.path.join(root, "ops", "hpu_fp8.py")
    init_py = os.path.join(root, "__init__.py")
    write(os.path.join(root, "ops", RT_NAME), rt_src)
    src = read(ops_py)
    if MARK not in src:
        write(ops_py, src.rstrip("\n") + "\n" + OPS_HOOK)
    src = read(init_py)
    if MARK not in src:
        write(init_py, INIT_HOOK + src)
    for p in (ops_py, init_py):
        if MARK not in read(p):
            sys.exit(f"{MARK} hook missing in {p} after install")
    print(f"{MARK} installed in {root} (runtime ops/{RT_NAME}; hooks in ops/hpu_fp8.py and __init__.py); "
          f"GLM_FP8_MXK={os.environ.get('GLM_FP8_MXK', '0')}")


if __name__ == "__main__":
    main()
