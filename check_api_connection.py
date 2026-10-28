"""读取项目 .env，实际测试 LLM 和向量接口；不使用已有缓存。"""

import argparse
from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import time

from memory_retrieval.api import APIClient, APIError
from memory_retrieval.config import PROJECT_ROOT, Settings


def check_llm(client):
    def validate(result):
        if result != {"status": "ok"}:
            raise ValueError('期望返回 {"status":"ok"}')
        return result

    client.json_call(
        "connection_check",
        '这是 API 连通性测试。请只返回 JSON：{"status":"ok"}。',
        {"message": "请确认能够正常响应。"},
        validate,
    )
    return '收到有效 JSON 回复：{"status":"ok"}'


def check_embedding(client):
    vectors = client.embed(["这是一条用于测试向量接口连通性的文本。"], "connection_check_embedding")
    return f"收到 1 条有效向量，维度为 {vectors.shape[1]}；数值、非零性校验通过"


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="真实请求测试 LLM 与向量接口，分别显示成功或失败")
    parser.add_argument("--env", type=Path, default=PROJECT_ROOT / ".env", help="环境文件路径，默认项目根目录 .env")
    parser.add_argument("--timeout", type=float, help="每次请求的超时秒数；默认使用 .env 的 API_TIMEOUT_SECONDS")
    args = parser.parse_args(argv)
    try:
        settings = Settings.from_env(args.env)
        if args.timeout is not None:
            if not 0 < args.timeout < float("inf"):
                raise ValueError("--timeout 必须为大于 0 的有限数字")
            settings = replace(settings, timeout=args.timeout)
    except (ValueError, OSError) as exc:
        print(f"[失败] 配置读取失败：{exc}")
        print("[结果] LLM 与向量测试均未执行。退出码：1。")
        return 1

    print("开始真实接口测试：不读取项目缓存，不打印 API Key。", flush=True)
    print(f"单次请求超时：{settings.timeout:g} 秒；网络错误不重试。", flush=True)
    results = []
    with tempfile.TemporaryDirectory(prefix="memory-api-check-") as temp:
        # 每次新建临时缓存，验证真实连接；退出时自动清理，不影响研究结果。
        client = APIClient(replace(settings, max_retries=0, cache_path=Path(temp) / "cache.sqlite3"))
        for label, check in (("LLM", check_llm), ("向量模型", check_embedding)):
            print(f"\n[测试] {label} ...", flush=True)
            start = time.perf_counter()
            try:
                detail = check(client)
            except (APIError, ValueError, OSError) as exc:
                detail = str(exc)
                for secret in (settings.llm_api_key, settings.embedding_api_key):
                    if secret:
                        detail = detail.replace(secret, "[已隐藏]")
                print(f"[失败] {label}（{time.perf_counter() - start:.2f} 秒）：{detail}", flush=True)
                results.append(False)
            else:
                print(f"[成功] {label}（{time.perf_counter() - start:.2f} 秒）：{detail}", flush=True)
                results.append(True)
    if all(results):
        print("\n[结果] 两项均成功：LLM 与向量接口均可连通，测试响应符合项目所需格式。")
        print("这里只验证接口，不代表完整研究流程或语义效果已通过真实模型验证。")
        return 0
    print(f"\n[结果] 存在失败：LLM={'成功' if results[0] else '失败'}，"
          f"向量模型={'成功' if results[1] else '失败'}。退出码：1。")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
