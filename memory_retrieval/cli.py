"""Evaluate write-time conflict retrieval in the selected memory banks."""

import argparse
from datetime import datetime, timezone
import math
from pathlib import Path
import sys

from .config import PROJECT_ROOT
from .data import read_jsonl
from .dataset import load_dataset
from .encoders import MODEL_ROOT
from .experiment import ABLATION, BASELINES, MAIN_METHOD, run
from .ocsp import ORTHO_GAMMA


def validate_translation(source, dataset):
    """The bilingual pilot uses identical IDs, metadata and labels in both languages."""
    if (source / "gold.jsonl").read_bytes() != (dataset / "gold.jsonl").read_bytes():
        raise ValueError("The two datasets have different gold labels")
    for filename in ("queries.jsonl", "memories.jsonl"):
        original, translated = read_jsonl(source / filename), read_jsonl(dataset / filename)
        for a, b in zip(original, translated, strict=True):
            if {k: v for k, v in a.items() if k != "content"} != {k: v for k, v in b.items() if k != "content"}:
                raise ValueError(f"Bilingual records differ outside content in {filename}")
            if not b["content"].strip() or any("\u4e00" <= c <= "\u9fff" for c in b["content"]):
                raise ValueError(f"Expected English content in {filename}")


def split_ids(values):
    if values is None:
        return None
    return [item.strip() for value in values for item in value.split(",") if item.strip()]


def run_cli(argv=None, *, default_methods=(MAIN_METHOD,)):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="输入数据集或单个 STALE 场景目录")
    parser.add_argument("--arms", nargs="+", choices=(*BASELINES, MAIN_METHOD, ABLATION, "all"),
                        default=default_methods, help=f"默认 {' '.join(default_methods)}。all 运行主方法和六个基线")
    select = parser.add_mutually_exclusive_group()
    select.add_argument("--scenario-ids", nargs="+", metavar="ID")
    select.add_argument("--query-ids", nargs="+", metavar="ID", help="支持空格或逗号分隔的查询 ID")
    parser.add_argument("--limit", type=int, default=0, help="随机抽取查询数，默认全部；指定 ID 时使用全部指定项")
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--output", type=Path, help="本次实验的新目录")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "runs/experiments")
    parser.add_argument("--run-name", help="在 --output-root 下使用的目录名，默认使用 UTC 时间戳")
    parser.add_argument("--env", type=Path, help="LLM 或可选向量 API 的 .env 文件")
    parser.add_argument("--cache", type=Path, help="可复用的探针响应缓存，默认在本次实验目录创建")
    parser.add_argument("--embedding-backend", choices=("bge", "api"), default="bge")
    parser.add_argument("--model-root", "--sparsecl-root", dest="model_root", type=Path, default=MODEL_ROOT,
                        help="包含冻结权重和官方 SparseCL 源码的本地目录")
    parser.add_argument("--gamma", type=float, default=ORTHO_GAMMA)
    for name in ("queries", "memories", "gold"):
        parser.add_argument(f"--{name}", type=Path, help=f"显式指定 {name}.jsonl")
    source_mode = parser.add_mutually_exclusive_group()
    source_mode.add_argument("--source", type=Path, help="可选，与英文翻译数据逐行核对的原文目录")
    source_mode.add_argument("--original-text", action="store_true", help="直接使用数据集正文（默认）")
    args = parser.parse_args(argv)
    if args.limit < 0:
        parser.error("--limit 必须大于等于 0")
    if not math.isfinite(args.gamma):
        parser.error("--gamma 必须是有限数")
    if args.output and args.run_name:
        parser.error("--output 和 --run-name 只能选一项")
    if args.run_name and (Path(args.run_name).name != args.run_name or args.run_name in (".", "..")):
        parser.error("--run-name 必须是目录名")
    methods = []
    for name in args.arms:
        methods.extend((MAIN_METHOD, *BASELINES) if name == "all" else (name,))
    methods = tuple(dict.fromkeys(methods))
    if args.embedding_backend == "api" and any(name.startswith("sparsecl_") for name in methods):
        parser.error("SparseCL 的初召回需要冻结 BGE，请使用 --embedding-backend bge")
    scenario_ids, query_ids = split_ids(args.scenario_ids), split_ids(args.query_ids)
    if scenario_ids == [] or query_ids == []:
        parser.error("至少指定一个有效 ID")
    if args.source:
        if args.queries or args.memories or args.gold:
            parser.error("--source 需要完整数据集目录")
        validate_translation(args.source, args.dataset)
    dataset = load_dataset(args.dataset, scenario_ids=scenario_ids, query_ids=query_ids, limit=args.limit,
                           seed=args.seed, queries_file=args.queries, memories_file=args.memories, gold_file=args.gold)
    name = args.run_name or datetime.now(timezone.utc).strftime("retrieval-%Y%m%dT%H%M%S%fZ")
    output = (args.output or args.output_root / name).resolve()
    args.model_root = args.model_root.resolve()
    return run(dataset, args, output, methods)


def main():
    run_cli()
