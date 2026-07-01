"""Configurable embedding clients for BrowseComp-Plus dense retrieval."""

from __future__ import annotations

import json
import math
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Iterable


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _as_int(value: Any, default: int) -> int:
    if value is None or value == "":
        return default
    return int(value)


def _as_float(value: Any, default: float) -> float:
    if value is None or value == "":
        return default
    return float(value)


def normalize_endpoint(endpoint: str) -> str:
    stripped = endpoint.rstrip("/")
    if stripped.endswith("/embeddings"):
        return stripped
    return f"{stripped}/embeddings"


def l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vector))
    if norm == 0:
        return vector
    return [x / norm for x in vector]


def enforce_dimensions(vector: list[float], dimensions: int) -> list[float]:
    if dimensions <= 0:
        return vector
    if len(vector) == dimensions:
        return vector
    if len(vector) > dimensions:
        return vector[:dimensions]
    raise ValueError(f"provider returned {len(vector)} dims, expected {dimensions}")


def request_batches(items: list[str], size: int, max_batch_chars: int) -> Iterable[list[str]]:
    batch: list[str] = []
    batch_chars = 0
    for item in items:
        item_chars = len(item)
        if batch and (len(batch) >= size or (max_batch_chars > 0 and batch_chars + item_chars > max_batch_chars)):
            yield batch
            batch = []
            batch_chars = 0
        batch.append(item)
        batch_chars += item_chars
    if batch:
        yield batch


@dataclass
class EmbeddingConfig:
    provider: str = "openai_compatible"
    endpoint: str = ""
    model: str = ""
    api_key: str = ""
    dimensions: int = 0
    normalize: bool = True
    query_prefix: str = ""
    passage_prefix: str = ""
    batch_size: int = 64
    max_text_chars: int = 0
    max_batch_chars: int = 0
    tokenizer_name: str = ""
    max_tokens: int = 0
    max_batch_tokens: int = 0
    timeout: float = 30.0
    max_retries: int = 2
    extra_headers: dict[str, str] = field(default_factory=dict)
    extra_body: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "EmbeddingConfig":
        raw = raw or {}
        return cls(
            provider=str(raw.get("provider") or "openai_compatible"),
            endpoint=str(raw.get("endpoint") or ""),
            model=str(raw.get("model") or ""),
            api_key=str(raw.get("api_key") or ""),
            dimensions=_as_int(raw.get("dimensions"), 0),
            normalize=_as_bool(raw.get("normalize"), True),
            query_prefix=str(raw.get("query_prefix") or ""),
            passage_prefix=str(raw.get("passage_prefix") or ""),
            batch_size=max(1, _as_int(raw.get("batch_size"), 64)),
            max_text_chars=max(0, _as_int(raw.get("max_text_chars"), 0)),
            max_batch_chars=max(0, _as_int(raw.get("max_batch_chars"), 0)),
            tokenizer_name=str(raw.get("tokenizer_name") or ""),
            max_tokens=max(0, _as_int(raw.get("max_tokens"), 0)),
            max_batch_tokens=max(0, _as_int(raw.get("max_batch_tokens"), 0)),
            timeout=_as_float(raw.get("timeout"), 30.0),
            max_retries=max(0, _as_int(raw.get("max_retries"), 2)),
            extra_headers=dict(raw.get("extra_headers") or {}),
            extra_body=dict(raw.get("extra_body") or {}),
        )

    def redacted(self) -> dict[str, Any]:
        data = {
            "provider": self.provider,
            "endpoint": self.endpoint,
            "model": self.model,
            "dimensions": self.dimensions,
            "normalize": self.normalize,
            "query_prefix": self.query_prefix,
            "passage_prefix": self.passage_prefix,
            "batch_size": self.batch_size,
            "max_text_chars": self.max_text_chars,
            "max_batch_chars": self.max_batch_chars,
            "tokenizer_name": self.tokenizer_name,
            "max_tokens": self.max_tokens,
            "max_batch_tokens": self.max_batch_tokens,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "extra_headers": dict(self.extra_headers),
            "extra_body": dict(self.extra_body),
        }
        if self.api_key:
            data["api_key"] = "***"
        return data


class BaseEmbeddingClient:
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self._tokenizer = None

    def embed_query(self, query: str) -> list[float]:
        return self.embed_texts([self._limit_text(self.config.query_prefix + query)])[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        prefixed = [self._limit_text(self.config.passage_prefix + text) for text in texts]
        return self.embed_texts(prefixed)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    def _limit_text(self, text: str) -> str:
        if self.config.max_text_chars > 0 and len(text) > self.config.max_text_chars:
            text = text[:self.config.max_text_chars]
        if self.config.max_tokens > 0:
            tokenizer = self._get_tokenizer()
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            if len(token_ids) > self.config.max_tokens:
                text = tokenizer.decode(token_ids[:self.config.max_tokens], skip_special_tokens=True)
        return text

    def _get_tokenizer(self):
        if self._tokenizer is None:
            if not self.config.tokenizer_name:
                raise ValueError("embedding.tokenizer_name is required when embedding.max_tokens is set")
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_name)
        return self._tokenizer


class OpenAICompatibleEmbeddingClient(BaseEmbeddingClient):
    """Embedding client for OpenAI-compatible /embeddings endpoints."""

    def __init__(self, config: EmbeddingConfig):
        super().__init__(config)
        if not config.endpoint:
            raise ValueError("embedding.endpoint is required for openai_compatible provider")
        if not config.model:
            raise ValueError("embedding.model is required for openai_compatible provider")
        self.url = normalize_endpoint(config.endpoint)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for batch in self._request_batches(texts):
            resp = self._call_embeddings(batch)
            vectors.extend(self._parse_response(resp))
        return self._post_process(vectors)

    def _request_batches(self, texts: list[str]) -> Iterable[list[str]]:
        if self.config.max_batch_tokens <= 0:
            yield from request_batches(texts, self.config.batch_size, self.config.max_batch_chars)
            return

        tokenizer = self._get_tokenizer()
        batch: list[str] = []
        batch_chars = 0
        batch_tokens = 0
        for text in texts:
            text_chars = len(text)
            text_tokens = len(tokenizer.encode(text, add_special_tokens=False))
            would_exceed_size = len(batch) >= self.config.batch_size
            would_exceed_chars = self.config.max_batch_chars > 0 and batch_chars + text_chars > self.config.max_batch_chars
            would_exceed_tokens = batch_tokens + text_tokens > self.config.max_batch_tokens
            if batch and (would_exceed_size or would_exceed_chars or would_exceed_tokens):
                yield batch
                batch = []
                batch_chars = 0
                batch_tokens = 0
            batch.append(text)
            batch_chars += text_chars
            batch_tokens += text_tokens
        if batch:
            yield batch

    def _call_embeddings(self, texts: list[str]) -> dict[str, Any]:
        body: dict[str, Any] = {
            "input": texts,
            "model": self.config.model,
        }
        body.update(self.config.extra_body)

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        headers.update(self.config.extra_headers)

        payload = json.dumps(body).encode("utf-8")
        last_error: Exception | None = None
        for attempt in range(1, self.config.max_retries + 2):
            req = urllib.request.Request(self.url, data=payload, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self.config.timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                last_error = exc
                response_body = exc.read().decode("utf-8", errors="replace")
                transient = exc.code >= 500 or exc.code == 429
                if not transient or attempt > self.config.max_retries:
                    raise RuntimeError(f"HTTP {exc.code} from embedding endpoint: {response_body}") from exc
            except (urllib.error.URLError, TimeoutError) as exc:
                last_error = exc
                if attempt > self.config.max_retries:
                    raise RuntimeError(f"network error calling embedding endpoint: {exc}") from exc

            time.sleep((0.2 * (2 ** (attempt - 1))) + random.random() * 0.1)

        raise RuntimeError(f"exhausted retries: {last_error}")

    def _parse_response(self, resp: dict[str, Any]) -> list[list[float]]:
        rows = resp.get("data")
        if not isinstance(rows, list):
            raise ValueError("embedding response missing data[]")

        if all(isinstance(row, dict) and "index" in row for row in rows):
            rows = sorted(rows, key=lambda row: int(row["index"]))

        vectors: list[list[float]] = []
        for i, row in enumerate(rows):
            if not isinstance(row, dict) or not isinstance(row.get("embedding"), list):
                raise ValueError(f"embedding response data[{i}] missing embedding[]")
            vectors.append([float(x) for x in row["embedding"]])
        return vectors

    def _post_process(self, raw_vectors: list[list[float]]) -> list[list[float]]:
        if not raw_vectors:
            return []

        inferred = len(raw_vectors[0]) if self.config.dimensions <= 0 else self.config.dimensions
        processed: list[list[float]] = []
        for i, vec in enumerate(raw_vectors):
            if self.config.dimensions <= 0 and len(vec) != inferred:
                raise ValueError(
                    f"inconsistent dimensions in embedding response: row 0 has {inferred}, row {i} has {len(vec)}"
                )
            out = enforce_dimensions(vec, inferred)
            if self.config.normalize:
                out = l2_normalize(out)
            processed.append(out)
        return processed


def create_embedding_client(raw_config: dict[str, Any] | EmbeddingConfig) -> BaseEmbeddingClient:
    config = raw_config if isinstance(raw_config, EmbeddingConfig) else EmbeddingConfig.from_dict(raw_config)
    provider = config.provider.strip().lower()
    if provider in {"openai_compatible", "openai-compatible", "memos_openai_compatible", "http"}:
        return OpenAICompatibleEmbeddingClient(config)
    raise ValueError(
        f"Unsupported embedding provider '{config.provider}'. "
        "Supported providers: openai_compatible"
    )
