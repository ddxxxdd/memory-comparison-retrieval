"""OpenAI 兼容 HTTP 接口；缓存请求/响应，不保存鉴权头。"""

from contextlib import closing
from http.client import HTTPException
import json
import sqlite3
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np

from .config import endpoint
from .data import canonical_json, fingerprint


class APIError(RuntimeError):
    pass


def unwrap_envelope(result):
    """部分 OpenAI 兼容网关会把标准响应再包一层 {"data": {...}, "success": ...}。
    仅当 data 是含 choices 的字典时剥离信封；embedding 的 data 是列表，不受影响。"""
    if not isinstance(result, dict):
        return result
    inner = result.get("data")
    if isinstance(inner, dict) and "choices" in inner:
        merged = dict(inner)
        if "usage" not in merged and isinstance(result.get("usage"), dict):
            merged["usage"] = result["usage"]
        return merged
    return result


class APIClient:
    def __init__(self, settings):
        self.settings = settings
        self.events = []
        settings.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(settings.cache_path, timeout=30)) as db, db:
            db.execute("CREATE TABLE IF NOT EXISTS responses (key TEXT PRIMARY KEY, request TEXT, response TEXT)")

    def _post(self, url, api_key, payload, stage):
        start = time.perf_counter()
        request_data = {"url": url, "body": payload}
        cache_key = fingerprint(request_data)
        with closing(sqlite3.connect(self.settings.cache_path, timeout=30)) as db, db:
            cached = db.execute("SELECT response FROM responses WHERE key = ?", (cache_key,)).fetchone()
        if cached:
            self.events.append({"stage": stage, "status": "ok", "cached": True,
                                "seconds": time.perf_counter() - start, "usage": None})
            return json.loads(cached[0])
        for attempt in range(self.settings.max_retries + 1):
            attempt_start = time.perf_counter()
            request = Request(url, data=canonical_json(payload).encode("utf-8"), method="POST", headers={
                "Authorization": f"Bearer {api_key}", "Content-Type": "application/json",
                # 部分网关(Cloudflare error 1010)按客户端指纹拦截默认 urllib,补一个常规 UA。
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                              "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
                "Accept": "application/json",
            })
            retry_after = None
            try:
                with urlopen(request, timeout=self.settings.timeout) as response:
                    result = json.load(response)
                result = unwrap_envelope(result)
                if not isinstance(result, dict) or "error" in result:
                    raise APIError(f"{stage}: 服务未返回正常的 JSON 对象")
            except HTTPError as exc:
                self.events.append({"stage": stage, "status": "http_error", "cached": False,
                                    "attempt": attempt + 1, "http_status": exc.code,
                                    "seconds": time.perf_counter() - attempt_start, "usage": None})
                # 不将远端错误正文写入日志或终端。
                code = exc.code
                header = exc.headers.get("Retry-After") if exc.headers else None
                exc.close()
                # 403 在本项目的向量网关上是可复现的瞬时拒绝（WAF/配额），会自行恢复；
                # 401 仍是鉴权错误，重试没有意义。
                retryable = code in {403, 408, 429} or code >= 500
                if not retryable or attempt == self.settings.max_retries:
                    if code == 402:
                        # 402 是上游账户余额不足，重试没有意义，但提示必须指向真正的原因。
                        hint = "上游账户余额不足，请充值或更换 Key"
                    elif code in {401, 403}:
                        hint = "鉴权失败或被网关限流"
                    else:
                        hint = "请检查服务地址、模型、Key 和 JSON mode 支持"
                    raise APIError(f"{stage}: HTTP {code};{hint}") from None
                # 429/403 是限流类拒绝,给更长的退避,否则并行查询会连续撞限。
                # 5xx 多为网关瞬时不稳,固定 4 秒退避会在同一故障上反复失败,
                # 因此同样按指数加长,上限 30 秒。
                if code in {403, 429}:
                    retry_after = self.settings.rate_limit_backoff * (attempt + 1)
                    if header and header.strip().isdigit():
                        retry_after = max(retry_after, float(header.strip()))
                elif code >= 500:
                    retry_after = min(2 ** attempt, 30)
            except (URLError, TimeoutError, OSError, HTTPException) as exc:
                # IncompleteRead 等 http.client 异常也是可重试的传输层故障；
                # 长跑任务里单次响应截断不应让整条查询失败。
                self.events.append({"stage": stage, "status": "network_error", "cached": False,
                                    "attempt": attempt + 1, "seconds": time.perf_counter() - attempt_start,
                                    "usage": None})
                if attempt == self.settings.max_retries:
                    raise APIError(f"{stage}: 网络连接失败或超时({type(exc).__name__})") from None
            except (ValueError, APIError):
                self.events.append({"stage": stage, "status": "invalid_response", "cached": False,
                                    "attempt": attempt + 1, "seconds": time.perf_counter() - attempt_start,
                                    "usage": None})
                raise APIError(f"{stage}: 服务响应不是有效 JSON,或含 error 字段") from None
            else:
                self.events.append({"stage": stage, "status": "ok", "cached": False,
                                    "attempt": attempt + 1, "seconds": time.perf_counter() - attempt_start,
                                    "usage": result.get("usage")})
                with closing(sqlite3.connect(self.settings.cache_path, timeout=30)) as db, db:
                    db.execute("INSERT OR REPLACE INTO responses VALUES (?, ?, ?)",
                               (cache_key, canonical_json(request_data), canonical_json(result)))
                return result
            time.sleep(min(2 ** attempt, 4) if retry_after is None else retry_after)
        raise APIError("请求未完成")

    def json_call(self, task, system, payload, validate, extra_messages=()):
        """extra_messages 追加在首轮消息之后，用于"上一次输出不合格，请重写"这类定向修复。"""
        self.settings.require(llm=True)
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": canonical_json({"task": task, **payload})},
                    *extra_messages]
        for repair in range(2):
            request = {"model": self.settings.llm_model, "messages": messages}
            if self.settings.llm_temperature is not None:
                request["temperature"] = self.settings.llm_temperature
            if self.settings.llm_json_mode:
                request["response_format"] = {"type": "json_object"}
            # 该网关的 flash 模型默认跑思维链,一次批处理会先烧掉数千 reasoning token;
            # 描述抽取不需要推理,显式关掉后同一请求从 40 秒级降到 2 秒级。
            if self.settings.llm_reasoning_effort:
                request["reasoning_effort"] = self.settings.llm_reasoning_effort
            result = self._post(endpoint(self.settings.llm_base_url, "chat/completions"),
                                self.settings.llm_api_key, request, task)
            try:
                choice = result["choices"][0]
                if choice.get("finish_reason") == "length":
                    raise APIError(f"{task}: 输出被服务截断，请调整服务端输出限制")
                raw = choice["message"]["content"]
                if not isinstance(raw, str):
                    raise APIError(f"{task}: 缺少文本输出")
            except (KeyError, IndexError, TypeError):
                raise APIError(f"{task}: 响应不符合 chat/completions 格式") from None
            try:
                text = raw.strip()
                if text.startswith("```") and text.endswith("```"):
                    text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
                value = json.loads(text)
                if not isinstance(value, dict):
                    raise ValueError("输出必须是 JSON 对象")
                return validate(value)
            except (ValueError, KeyError, TypeError, IndexError) as exc:
                self.events.append({"stage": task, "status": "invalid_output",
                                    "cached": None, "seconds": 0, "usage": None,
                                    "detail": str(exc), "repair": repair})
                if repair:
                    raise APIError(f"{task}: 一次格式修复后仍无效：{exc}") from None
                messages.extend([
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": f"上一次输出不符合约定：{exc}。仅修复输出，返回完整合法 JSON。"},
                ])

    def embed(self, texts, stage="embedding"):
        self.settings.require(embedding=True)
        if not texts:
            raise ValueError("不能编码空文本列表")
        vectors = []
        for offset in range(0, len(texts), self.settings.embedding_batch_size):
            batch = texts[offset:offset + self.settings.embedding_batch_size]
            payload = {"model": self.settings.embedding_model, "input": batch, "encoding_format": "float"}
            result = self._post(endpoint(self.settings.embedding_base_url, "embeddings"),
                                self.settings.embedding_api_key, payload, stage)
            try:
                items = result["data"]
                ids = [item["index"] for item in items]
                if any(type(i) is not int for i in ids) or sorted(ids) != list(range(len(batch))):
                    raise ValueError("向量响应 index 缺失或重复")
                vectors.extend(item["embedding"] for item in sorted(items, key=lambda item: item["index"]))
            except (KeyError, TypeError, ValueError):
                raise APIError(f"{stage}: 响应不符合 embeddings 格式或数量不一致") from None
        try:
            matrix = np.asarray(vectors, dtype=np.float32)
            if matrix.ndim != 2 or matrix.shape[1] == 0 or not np.isfinite(matrix).all():
                raise ValueError()
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            if not np.isfinite(norms).all() or np.any(norms == 0):
                raise ValueError()
            return matrix / norms
        except (ValueError, TypeError):
            raise APIError(f"{stage}: 向量必须为同维、有限、非零数值") from None


def usage_summary(events):
    network = [e for e in events if e["cached"] is False]
    successful = [e for e in network if e["status"] == "ok"]
    known = [e["usage"] for e in successful if isinstance(e.get("usage"), dict)]
    return {
        "api_attempts": len(network),
        "cache_hits": sum(e["cached"] is True for e in events),
        "invalid_outputs": sum(e["status"] == "invalid_output" for e in events),
        "known_prompt_tokens": sum(u.get("prompt_tokens") or 0 for u in known),
        "known_completion_tokens": sum(u.get("completion_tokens") or 0 for u in known),
        "known_total_tokens": sum(u.get("total_tokens") or 0 for u in known),
        "responses_without_usage": len(successful) - len(known),
        "cost_currency": None,
    }
