import argparse
import json
from pathlib import Path
import sys

from .adapters import load_graph, load_text
from .api import APIClient, APIError
from .config import DESCRIPTION_DUPLICATE_THRESHOLD, MEMORY_TEMPLATE_THRESHOLD, Settings
from .data import build_retrieval_text, canonical_json, load_queries, read_jsonl, write_json
from .evaluation import evaluate, validate_gold
from .index import MemoryIndex, build_index
from .pipeline import Retriever


def parser():
    root = argparse.ArgumentParser(description="固定旧库的可比较记忆检索（先选已有描述，再生成未覆盖事项）")
    root.add_argument("--env", type=Path, help=".env 路径，默认项目根目录的 .env")
    sub = root.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="离线适配、抽取描述、建立冻结索引")
    build.add_argument("--input", type=Path, required=True)
    build.add_argument("--adapter", choices=["text", "graph"], default="text")
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--description-mode", choices=["independent", "reuse", "batch_reuse"],
                       default="batch_reuse",
                       help="默认批式复用：一批记忆共用一次请求，先引用已有描述，批内同一事项收敛到同一份文本")
    build.add_argument("--batch-size", type=int, default=16, help="batch_reuse 每批处理的记忆条数")
    build.add_argument("--no-repair", action="store_true",
                       help="关掉抽取校验后的重写（M1 消融；仍会记录违规，但不改输出）")
    build.add_argument("--consolidate", action="store_true",
                       help="开启建库后的目录归并（M2；默认关闭，实测会降低候选召回）")
    query = sub.add_parser("query", help="每条新记忆独立检索同一个冻结旧库")
    query.add_argument("--index", type=Path, required=True)
    query.add_argument("--queries", type=Path, required=True)
    query.add_argument("--output", type=Path, required=True)
    query.add_argument("--method", choices=["proposed", "independent", "dense", "bm25", "hybrid"], default="proposed")
    query.add_argument("--no-rerank", action="store_true", help="描述检索的无假设句重排消融")
    query.add_argument("--judge", action="store_true", help="候选输出后运行固定逐对冲突判定器")
    query.add_argument("--coarse-k", type=int, help="top-k 对照的粗候选预算")
    query.add_argument("--final-k", type=int, help="top-k 对照的最终候选预算")
    metric = sub.add_parser("evaluate", help="计算四项实验指标")
    metric.add_argument("--gold", type=Path, required=True)
    metric.add_argument("--results", type=Path, required=True)
    metric.add_argument("--output", type=Path, required=True)
    check = sub.add_parser("validate-data", help="不调用模型，检查输入与金标引用")
    check.add_argument("--memories", type=Path, required=True)
    check.add_argument("--adapter", choices=["text", "graph"], default="text")
    check.add_argument("--queries", type=Path)
    check.add_argument("--gold", type=Path)
    return root


def main(argv=None):
    # Windows 重定向输出也保持 UTF-8，与输入/输出文件的编码一致。
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    args = parser().parse_args(argv)
    client = None
    try:
        if args.command == "validate-data":
            memories = (load_text if args.adapter == "text" else load_graph)(args.memories)
            queries = load_queries(args.queries) if args.queries else []
            gold = validate_gold(read_jsonl(args.gold), {m.memory_id for m in memories}) if args.gold else []
            if gold and queries and {q["query_id"] for q in queries} != {g["query_id"] for g in gold}:
                raise ValueError("查询与金标 query_id 不一致")
            print(f"输入有效：{len(memories)} 条旧记忆，{len(queries)} 条查询，{len(gold)} 条金标。")
            return
        if args.command == "evaluate":
            if args.output.exists():
                raise ValueError("输出文件已存在，请使用新的输出路径")
            report = evaluate(read_jsonl(args.gold), read_jsonl(args.results))
            write_json(args.output, report)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return
        settings = Settings.from_env(args.env)
        if args.command == "build":
            settings.require(llm=True, embedding=True)
            if args.output.exists():
                raise ValueError("索引目录已存在，请使用新目录")
            memories = (load_text if args.adapter == "text" else load_graph)(args.input)
            client = APIClient(settings)
            index = build_index(memories, client, args.output, args.description_mode,
                                batch_size=args.batch_size,
                                description_threshold=(DESCRIPTION_DUPLICATE_THRESHOLD
                                                       if args.consolidate else None),
                                template_threshold=(MEMORY_TEMPLATE_THRESHOLD
                                                    if args.consolidate else None),
                                repair_rounds=0 if args.no_repair else 1)
            print(f"索引已冻结：{len(index.memories)} 条旧记忆，{len(index.catalog)} 个描述入口。")
            print(f"描述模式：{index.manifest['description_mode']}"
                  + (f"，每批 {index.manifest['batch_size']} 条" if index.manifest.get("batch_size") else ""))
            print(f"目录：{args.output.resolve()}")
            return
        index = MemoryIndex.load(args.index)
        queries = load_queries(args.queries)
        if args.output.exists():
            raise ValueError("结果文件已存在，请使用新的输出路径")
        settings.require(llm=args.method in {"proposed", "independent"} or args.judge,
                         embedding=args.method != "bm25")
        if args.method != "bm25":
            index.check_embedding(settings)
        if args.method not in {"proposed", "independent"}:
            settings.validate_budget(args.coarse_k if args.coarse_k is not None else settings.coarse_k,
                                     args.final_k if args.final_k is not None else settings.final_k)
        client = APIClient(settings)
        retriever = Retriever(index, client)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # 逐查询保存，长批次失败时保留已完成记录；再次执行可利用 API 缓存。
        with args.output.open("x", encoding="utf-8") as stream:
            for query in queries:
                result = retriever.query(
                    query["query_id"], query["content"],
                    retrieval_text=build_retrieval_text(query["content"], query.get("source_ref")),
                    method=args.method, rerank=not args.no_rerank, judge=args.judge,
                    coarse_k=args.coarse_k, final_k=args.final_k)
                stream.write(canonical_json(result) + "\n")
                stream.flush()
                print(f"{query['query_id']}: {len(result['coarse_candidates'])} 条粗候选 → {len(result['candidates'])} 条候选")
        print(f"结果：{args.output.resolve()}")
    except (ValueError, APIError, KeyError, OSError, TypeError) as exc:
        # 失败调用也留下可审查的调用统计；不写 API Key 或 HTTP 错误正文。
        if client is not None:
            error_path = client.settings.cache_path.parent / "last_error.json"
            write_json(error_path, {"error": str(exc), "events": client.events})
        print(f"错误：{exc}", file=sys.stderr)
        raise SystemExit(1) from None
