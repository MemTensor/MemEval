#!/usr/bin/env python3
"""Start the MCP search server for BrowseComp-Plus.

Reads mcp_server section from information_retrieval.yaml and launches the
FAISS search server.

Usage:
    python start_mcp.py                          # default config
    python start_mcp.py /path/to/custom.yaml     # custom config
"""

import json
import os
import sys
import glob
from typing import Any

import yaml

_DIR = os.path.dirname(os.path.abspath(__file__))
_SEARCHER_DIR = os.path.join(_DIR, "searcher")
_PROJECT_ROOT = os.path.abspath(os.path.join(_DIR, "..", "..", "..", ".."))


def _load_project_dotenv() -> None:
    env_path = os.path.join(_PROJECT_ROOT, ".env.agent")
    if not os.path.exists(env_path):
        return
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                os.environ.setdefault(key.strip(), val.strip())


def _resolve_env_vars(obj: Any) -> Any:
    if isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
        return os.environ.get(obj[2:-1], "")
    if isinstance(obj, dict):
        return {k: _resolve_env_vars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_env_vars(v) for v in obj]
    return obj


def _resolve_local_path(value: str, config_dir: str) -> str:
    if not value or os.path.isabs(value) or "://" in value:
        return value
    project_resolved = os.path.join(_PROJECT_ROOT, value)
    if value.startswith(".") and (
        glob.glob(project_resolved) or os.path.exists(project_resolved.split("*")[0].rstrip("/"))
    ):
        return os.path.abspath(project_resolved)
    resolved = os.path.join(config_dir, value)
    if value.startswith(".") or os.path.exists(resolved.split("*")[0].rstrip("/")):
        return os.path.abspath(resolved)
    return value


def _add_cli_arg(argv: list[str], key: str, value: Any) -> None:
    if value is None or value == "":
        return
    flag = "--" + key.replace("_", "-")
    if isinstance(value, bool):
        if value:
            argv.append(flag)
        return
    argv.extend([flag, str(value)])


def _searcher_arg_keys(searcher_type: str) -> list[str]:
    faiss_keys = [
        "model_name",
        "normalize",
        "pooling",
        "torch_dtype",
        "dataset_name",
        "task_prefix",
        "max_length",
    ]
    configurable_keys = [
        "index_metadata",
        "metric",
        "dataset_name",
        "dataset_split",
        "corpus_file",
        "docid_field",
        "text_field",
    ]
    return {
        "faiss": faiss_keys,
        "reasonir": faiss_keys,
        "configurable_faiss": configurable_keys,
        "bm25": [],
        "custom": [],
    }.get(searcher_type, [])


def main():
    _load_project_dotenv()

    # Auto-detect JDK from conda env (pyserini/jnius needs JAVA_HOME + JVM_PATH)
    prefix = sys.prefix
    if not os.environ.get("JAVA_HOME"):
        if os.path.exists(os.path.join(prefix, "bin", "java")):
            os.environ["JAVA_HOME"] = prefix
    if not os.environ.get("JVM_PATH"):
        jvm = os.path.join(prefix, "lib", "jvm", "lib", "server", "libjvm.so")
        if os.path.exists(jvm):
            os.environ["JVM_PATH"] = jvm

    _default_yaml = os.path.join(_DIR, "..", "..", "domains", "information_retrieval", "information_retrieval.yaml")
    config_path = sys.argv[1] if len(sys.argv) > 1 else _default_yaml
    with open(config_path) as f:
        root_cfg = _resolve_env_vars(yaml.safe_load(f) or {})
        cfg = root_cfg["mcp_server"]

    # Resolve relative paths against yaml file directory
    config_dir = os.path.dirname(os.path.abspath(config_path))
    for key in ("index_path", "index_metadata", "corpus_file", "model_name"):
        val = cfg.get(key, "")
        if not isinstance(val, str):
            continue
        cfg[key] = _resolve_local_path(val, config_dir)

    # Make BrowseComp-Plus searcher importable
    if _SEARCHER_DIR not in sys.path:
        sys.path.insert(0, _SEARCHER_DIR)

    searcher_type = cfg.get("searcher_type", "faiss")
    argv = [
        "mcp_server",
        "--searcher-type", searcher_type,
        "--index-path", cfg["index_path"],
        "--transport", "sse",
        "--port", str(cfg.get("port", 9100)),
        "--k", str(cfg.get("k", 5)),
        "--snippet-max-tokens", str(cfg.get("snippet_max_tokens", 512)),
    ]
    if cfg.get("get_document"):
        argv.append("--get-document")

    for key in _searcher_arg_keys(searcher_type):
        _add_cli_arg(argv, key, cfg.get(key))

    if searcher_type == "configurable_faiss":
        embedding_cfg = dict(root_cfg.get("embedding") or {})
        embedding_cfg.update(cfg.get("embedding") or {})
        if not embedding_cfg:
            raise ValueError("embedding config is required for searcher_type=configurable_faiss")
        argv.extend(["--embedding-config-json", json.dumps(embedding_cfg, ensure_ascii=False)])

    sys.argv = argv

    from mcp_server import main as mcp_main
    mcp_main()


if __name__ == "__main__":
    main()
