from dataclasses import dataclass
import os
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import dotenv_values


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SIMILARITY_THRESHOLD = 0.7

# 建库阶段的三个固定先验阈值，只在"判定两个东西是不是同一事项"时使用，
# 不参与查询打分，也不按评测结果调整。
#   两条记忆内容的相似度达到此值，视为"同一事项的不同取值"，必须共用描述；
MEMORY_TEMPLATE_THRESHOLD = 0.90
#   两条描述文本的相似度达到此值，视为同一事项的不同措辞，必须收成一条；
DESCRIPTION_DUPLICATE_THRESHOLD = 0.95


@dataclass(frozen=True)
class Settings:
    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_model: str = ""
    embedding_base_url: str = ""
    embedding_api_key: str = ""
    embedding_model: str = ""
    llm_json_mode: bool = True
    llm_temperature: float | None = 0.0
    llm_reasoning_effort: str | None = None
    timeout: float = 120.0
    max_retries: int = 4
    rate_limit_backoff: float = 8.0
    embedding_batch_size: int = 32
    coarse_k: int = 50
    final_k: int = 10
    cache_path: Path = PROJECT_ROOT / "artifacts/api_cache.sqlite3"

    @classmethod
    def from_env(cls, path: Path | None = None):
        env_path = path or PROJECT_ROOT / ".env"
        if path is not None and not env_path.is_file():
            raise ValueError(f"指定的环境文件不存在：{env_path}")
        values = {**dotenv_values(env_path, encoding="utf-8-sig"), **os.environ}

        def value(name, default=""):
            return (values.get(name) or default).strip()

        mode = value("LLM_JSON_MODE", "true").lower()
        if mode not in {"true", "false"}:
            raise ValueError("LLM_JSON_MODE 必须为 true 或 false")
        temp = value("LLM_TEMPERATURE")
        reasoning_effort = value("LLM_REASONING_EFFORT").lower() or None
        if reasoning_effort is not None and reasoning_effort not in {
                "none", "minimal", "low", "medium", "high"}:
            raise ValueError("LLM_REASONING_EFFORT 只能是 none/minimal/low/medium/high 或留空")
        cache_path = Path(value("CACHE_PATH", "artifacts/api_cache.sqlite3"))
        if not cache_path.is_absolute():
            cache_path = PROJECT_ROOT / cache_path
        settings = cls(
            llm_base_url=value("LLM_BASE_URL"),
            llm_api_key=value("LLM_API_KEY"),
            llm_model=value("LLM_MODEL"),
            embedding_base_url=value("EMBEDDING_BASE_URL"),
            embedding_api_key=value("EMBEDDING_API_KEY"),
            embedding_model=value("EMBEDDING_MODEL"),
            llm_json_mode=mode == "true",
            llm_temperature=float(temp) if temp else None,
            llm_reasoning_effort=reasoning_effort,
            timeout=float(value("API_TIMEOUT_SECONDS", "120")),
            max_retries=int(value("API_MAX_RETRIES", "4")),
            rate_limit_backoff=float(value("API_RATE_LIMIT_BACKOFF", "8")),
            embedding_batch_size=int(value("EMBEDDING_BATCH_SIZE", "32")),
            coarse_k=int(value("COARSE_K", "50")),
            final_k=int(value("FINAL_K", "10")),
            cache_path=cache_path,
        )
        if settings.timeout <= 0 or settings.max_retries < 0:
            raise ValueError("超时必须大于 0,重试次数不能为负")
        if settings.rate_limit_backoff < 0:
            raise ValueError("API_RATE_LIMIT_BACKOFF 不能为负")
        if settings.embedding_batch_size < 1:
            raise ValueError("EMBEDDING_BATCH_SIZE 必须大于 0")
        settings.validate_budget(settings.coarse_k, settings.final_k)
        return settings

    @staticmethod
    def validate_budget(coarse_k, final_k):
        if not 1 <= final_k <= coarse_k:
            raise ValueError("候选预算必须满足 1 <= FINAL_K <= COARSE_K")

    def require(self, *, llm=False, embedding=False):
        for prefix, needed in (("llm", llm), ("embedding", embedding)):
            if not needed:
                continue
            for suffix in ("base_url", "api_key", "model"):
                if not getattr(self, f"{prefix}_{suffix}"):
                    raise ValueError(f"请先在 .env 填写 {prefix.upper()}_{suffix.upper()}")
            url = urlsplit(getattr(self, f"{prefix}_base_url"))
            if url.scheme not in {"http", "https"} or not url.netloc:
                raise ValueError(f"{prefix.upper()}_BASE_URL 必须是 HTTP(S) 地址")
            if url.username or url.password or url.query or url.fragment:
                raise ValueError("BASE_URL 不应包含账号、密码、查询参数或片段；Key 请填入独立配置项")

    def embedding_identity(self):
        return {"url": endpoint(self.embedding_base_url, "embeddings"), "model": self.embedding_model}

    def llm_identity(self):
        return {
            "url": endpoint(self.llm_base_url, "chat/completions"),
            "model": self.llm_model,
            "temperature": self.llm_temperature,
            "json_mode": self.llm_json_mode,
            # 关掉思维链会改变调用语义,所以必须进索引身份,避免两次建库被当成同一配置。
            "reasoning_effort": self.llm_reasoning_effort,
        }


def endpoint(base, route):
    base = base.rstrip("/")
    return base if base.endswith("/" + route) else base + "/" + route
