# grid_search.py
from __future__ import annotations

import argparse
import itertools
import logging
import re
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import List

# ---------------------------------------------------------------------------
# 默认搜索空间 --------------------------------------------------------------
DEFAULT_MODEL_NAMES:   List[str] = ["PASTNet"]            # ←① 多模型遍历
DEFAULT_DATA_NAMES:    List[str] = ["sd", "gba", "gla", "ca"]
DEFAULT_INPUT_LENS:    List[int]  = [96]                  # ←② 输入长度搜索
DEFAULT_OUTPUT_LENS:   List[int]  = [48]                  # ←③ 输出长度搜索
DEFAULT_BATCH_SIZES:   List[int]  = [32]                  # ←④ 批次大小搜索

DEFAULT_GPU_ID = "2"
DEFAULT_CONFIG_NAME = "config"
DEFAULT_KEEP_TEMP = False

NUM_NODES_MAP: dict[str, int] = {
    "CA": 8600,
    "GBA": 2352,
    "GLA": 3834,
    "SD": 716,
}
# ---------------------------------------------------------------------------


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。"""
    p = argparse.ArgumentParser(description="网格超参数搜索")
    p.add_argument("--model-names",  nargs="+", default=DEFAULT_MODEL_NAMES,  metavar="MODEL")
    p.add_argument("--data-names",   nargs="+", default=DEFAULT_DATA_NAMES,  metavar="DATA")
    p.add_argument("--input-lens",   nargs="+", type=int, default=DEFAULT_INPUT_LENS,  metavar="T_in")
    p.add_argument("--output-lens",  nargs="+", type=int, default=DEFAULT_OUTPUT_LENS, metavar="T_out")
    p.add_argument("--batch-sizes",  nargs="+", type=int, default=DEFAULT_BATCH_SIZES, metavar="BS")
    p.add_argument("--gpu", "-g",    default=DEFAULT_GPU_ID, metavar="ID")
    p.add_argument("--config-name",  default=DEFAULT_CONFIG_NAME, metavar="NAME")
    p.add_argument("--keep-temp",    action="store_true", default=DEFAULT_KEEP_TEMP)
    return p.parse_args(argv)


def read_template(path: Path) -> str:
    """读取模板文本，若不存在则抛出 FileNotFoundError。"""
    if not path.exists():
        raise FileNotFoundError(f"Template not found: {path}")
    return path.read_text(encoding="utf-8")


def patch_template(
    *,
    src: str,
    data_name: str,
    input_len: int,
    output_len: int,
    batch_size: int,
) -> str:
    """将超参数注入模板源码。"""
    num_nodes = NUM_NODES_MAP.get(data_name)
    if num_nodes is None:
        raise ValueError(f"Unknown DATA_NAME {data_name!r}")

    # 清理旧定义
    cleaned_src = re.sub(
        r"^(DATA_NAME|INPUT_LEN|OUTPUT_LEN|BATCH_SIZE|num_nodes)\s*=.*?$",
        "",
        src,
        flags=re.M,
    ).lstrip()

    header = textwrap.dedent(
        f"""
        # ======== Auto-generated section (DO NOT EDIT) ========
        DATA_NAME   = "{data_name}"
        num_nodes   = {num_nodes}
        INPUT_LEN   = {input_len}
        OUTPUT_LEN  = {output_len}
        BATCH_SIZE  = {batch_size}
        # ======================================================
        """
    ).strip() + "\n\n"

    return header + cleaned_src


def _delete_pycache(pkg_dir: Path, module_basename: str) -> None:
    """删除指定模块的 .pyc 缓存。"""
    pycache_dir = pkg_dir / "__pycache__"
    if not pycache_dir.exists():
        return
    for pyc in pycache_dir.glob(f"{module_basename}.*.pyc"):
        pyc.unlink(missing_ok=True)


def _launch_training(cfg_module: str, gpu: str) -> None:
    """调用训练脚本。"""
    cmd = [sys.executable, "experiments/train.py", "-g", gpu, "-c", cfg_module]
    logging.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 外层先遍历模型名
    for model_name in args.model_names:
        pkg_dir = Path("baselines") / model_name
        tpl_path = pkg_dir / f"{args.config_name}.py"
        template_src = read_template(tpl_path)

        # 内层笛卡尔积遍历其余参数
        for data_name, t_in, t_out, bs in itertools.product(
            args.data_names,
            args.input_lens,
            args.output_lens,
            args.batch_sizes,
        ):
            module_basename = (
                f"{args.config_name}_{model_name}_{data_name}_{t_in}_{t_out}_{bs}"
            )
            cfg_path = pkg_dir / f"{module_basename}.py"

            # 1) 生成临时 config
            cfg_source = patch_template(
                src=template_src,
                data_name=data_name,
                input_len=t_in,
                output_len=t_out,
                batch_size=bs,
            )
            cfg_path.write_text(cfg_source, encoding="utf-8")

            # 2) 训练
            cfg_module = f"baselines.{model_name}.{module_basename}"
            try:
                _launch_training(cfg_module, args.gpu)
            finally:
                # 3) 清理
                if not args.keep_temp:
                    cfg_path.unlink(missing_ok=True)
                    _delete_pycache(pkg_dir, module_basename)


if __name__ == "__main__":
    main()
