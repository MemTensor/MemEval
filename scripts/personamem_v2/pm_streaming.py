#!/usr/bin/env python3
"""PersonaMem v2 streaming add-search-delete pipeline.

Each persona is treated as one independent evaluation unit:

1. add that persona's chat history;
2. search all benchmark questions for that persona;
3. save PersonaMem-compatible search results;
4. delete the persona/user from the memory service.

For Graphiti, chat histories are packed into bounded raw episodes with
chunk-level checkpoints. The combined output is compatible with
pm_responses.py, pm_metric.py, and pm_report.py.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent

sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

from client_factory import DEFAULT_LIB, SUPPORTED_LIBS, create_client, iter_batches  # noqa: E402
from personamem_v2.pm_ingestion import (  # noqa: E402
    CHAT_HISTORY_DIR,
    BENCHMARK_CSV,
    build_chat_history_index,
    ingest_session,
    load_benchmark_personas,
)
from personamem_v2.pm_search import (  # noqa: E402
    build_options,
    load_benchmark_rows,
    parse_user_query,
)
from personamem_v2.pm_common import (  # noqa: E402
    STATUS_SKIPPED,
    STATUS_SUCCESS_EMPTY,
    build_search_entry,
    classify_search_status,
    error_payload,
    status_counts,
)
from utils.checkpoint import atomic_json_dump  # noqa: E402
from utils.duration_stats import update_unit_duration_list  # noqa: E402
from utils.env import load_env  # noqa: E402
from utils.ingest_helpers import AddCallTimer  # noqa: E402
from utils.response_options import parse_bool  # noqa: E402
from utils.search_helpers import dispatch_search, unpack_search_result  # noqa: E402
from utils.streaming import (  # noqa: E402
    configure_single_user_streaming,
    delete_user_data,
    load_marker_set as load_added_chunks,
    load_marker_set as load_completed,
    log_event,
    LongCallLogger,
    mark_marker as mark_added_chunk,
    mark_marker as mark_completed,
    prepare_user_after_delete,
    resolve_max_batch_chars,
)

@dataclass(frozen=True)
class PersonaGraphitiChunk:
    persona_id: int
    part_idx: int
    total_parts: int
    content: str
    uuid: str

    @property
    def chunk_id(self) -> str:
        return f"{self.persona_id}_{self.part_idx}"


def _results_dir(frame: str, version: str) -> Path:
    return Path("results") / "pmv2" / f"{frame}-{version}"


def _row_tmp_path(frame: str, version: str, row_idx: int) -> Path:
    return _results_dir(frame, version) / "tmp" / f"{frame}_pm_search_results_{row_idx}.json"


def _combined_path(frame: str, version: str) -> Path:
    return _results_dir(frame, version) / f"{frame}_pm_search_results.json"


def _completed_path(frame: str, version: str) -> Path:
    return _results_dir(frame, version) / "streaming_completed.txt"


def _events_path(frame: str, version: str) -> Path:
    return _results_dir(frame, version) / "streaming_events.jsonl"


def _stats_path(frame: str, version: str) -> Path:
    return _results_dir(frame, version) / f"{frame}_pm_ingestion_stats.json"


def _added_chunks_path(frame: str, version: str, persona_id: int) -> Path:
    return _results_dir(frame, version) / "tmp" / f"{frame}_pm_added_chunks_{persona_id}.txt"


def user_id_for(version: str, persona_id: int) -> str:
    return f"pm_exper_user_{persona_id}_{version}"


def result_key_for(version: str, row_idx: int) -> str:
    return f"pm_exper_user_{row_idx}_{version}"


def _stable_uuid(*parts: object) -> str:
    key = ":".join(str(part) for part in parts)
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"omnimemeval:{key}"))


def _split_with_header(body: str, max_chars: int, header_fn) -> list[str]:
    if max_chars <= 0:
        return [header_fn(1, 1) + body]

    probe = header_fn(9999, 9999)
    body_limit = max(1, max_chars - len(probe) - 20)
    pieces: list[str] = []
    start = 0
    while start < len(body):
        end = min(start + body_limit, len(body))
        if end < len(body):
            newline = body.rfind("\n", start + 1, end)
            if newline > start + body_limit // 2:
                end = newline + 1
        pieces.append(body[start:end])
        start = end

    total = len(pieces)
    return [header_fn(idx, total) + piece for idx, piece in enumerate(pieces, start=1)]


def build_graphiti_persona_chunks(
    persona_id: int,
    chat_history: list[dict],
    *,
    version: str,
    max_chars: int,
) -> list[PersonaGraphitiChunk]:
    body_lines = [
        "PERSONAMEM V2 CHAT HISTORY",
        f"PERSONA ID: {persona_id}",
        "Use TURN labels to preserve conversation order.",
        "",
    ]
    message_count = 0
    for turn_idx, msg in enumerate(chat_history):
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system" or not content.strip():
            continue
        body_lines.append(f"PERSONA {persona_id} TURN {turn_idx} {role}: {content}")
        message_count += 1
    if message_count == 0:
        return []

    body = "\n".join(body_lines)

    def header(part_idx: int, total_parts: int) -> str:
        return (
            "PERSONAMEM V2 CHAT HISTORY CHUNK\n"
            f"persona={persona_id}; part={part_idx}/{total_parts}.\n"
            "Preserve original PERSONA/TURN labels and part order.\n"
        )

    contents = _split_with_header(body, max_chars, header)
    total = len(contents)
    return [
        PersonaGraphitiChunk(
            persona_id=persona_id,
            part_idx=idx,
            total_parts=total,
            content=content,
            uuid=_stable_uuid("pm", version, persona_id, idx),
        )
        for idx, content in enumerate(contents, start=1)
    ]


def add_persona_graphiti_chunked(
    persona_id: int,
    chat_history: list[dict],
    version: str,
    client,
    *,
    max_chars: int,
    added_chunks_path: Path,
    added_chunks: set[str],
) -> tuple[list[float], dict]:
    user_id = user_id_for(version, persona_id)
    chunks = build_graphiti_persona_chunks(
        persona_id,
        chat_history,
        version=version,
        max_chars=max_chars,
    )
    timer = AddCallTimer(client)
    skipped_chunks = 0

    for chunk in chunks:
        if chunk.chunk_id in added_chunks:
            skipped_chunks += 1
            continue
        session_key = (
            f"{user_id}_pm_history_part_{chunk.part_idx:04d}_"
            f"of_{chunk.total_parts:04d}"
        )
        label = (
            f"graphiti PersonaMem add persona={persona_id} "
            f"part={chunk.part_idx}/{chunk.total_parts} chars={len(chunk.content)}"
        )
        with LongCallLogger(label):
            client.add(
                [],
                user_id,
                session_key=session_key,
                raw_content=chunk.content,
                role="personamem_chat_history_chunk",
                source_description=(
                    f"PersonaMem v2 chat history chunk persona={persona_id} "
                    f"part={chunk.part_idx}/{chunk.total_parts}"
                ),
            )
        mark_added_chunk(added_chunks_path, added_chunks, chunk.chunk_id)

    msg_count = sum(
        1
        for msg in chat_history
        if msg.get("role") != "system" and msg.get("content", "").strip()
    )
    char_count = sum(
        len(msg.get("content", ""))
        for msg in chat_history
        if msg.get("role") != "system" and msg.get("content", "").strip()
    )
    meta = {
        "messages": msg_count,
        "chars": char_count,
        "chunks": len(chunks),
        "skipped_chunks": skipped_chunks,
        "max_chars": max_chars,
    }
    print(
        f"[graphiti] Persona {persona_id}: {msg_count} messages, "
        f"{char_count} chars, {len(chunks)} chunks ({skipped_chunks} resumed)"
    )
    return timer.durations_ms, meta


def add_persona_per_chunk(
    persona_id: int,
    chat_history: list[dict],
    frame: str,
    version: str,
    client,
    *,
    added_chunks_path: Path,
    added_chunks: set[str],
) -> tuple[list[float], dict]:
    user_id = user_id_for(version, persona_id)
    timer = AddCallTimer(client)
    chunks = 0
    skipped_chunks = 0
    for chunk_idx, chunk in enumerate(iter_batches(chat_history)):
        chunks += 1
        checkpoint_id = f"{persona_id}_chunk_{chunk_idx:03d}"
        if checkpoint_id in added_chunks:
            skipped_chunks += 1
            print(
                f"[{frame}] Persona {persona_id}: chunk {chunk_idx + 1} "
                "already added, skipping",
                flush=True,
            )
            continue
        session_id = f"pm_persona_{persona_id}_chunk_{chunk_idx}"
        ingest_session(chunk, user_id, session_id, frame, client)
        mark_added_chunk(added_chunks_path, added_chunks, checkpoint_id)
    msg_count = sum(
        1
        for msg in chat_history
        if msg.get("role") != "system" and msg.get("content", "").strip()
    )
    char_count = sum(
        len(msg.get("content", ""))
        for msg in chat_history
        if msg.get("role") != "system" and msg.get("content", "").strip()
    )
    return timer.durations_ms, {
        "messages": msg_count,
        "chars": char_count,
        "chunks": chunks,
        "skipped_chunks": skipped_chunks,
        "max_chars": None,
    }


def group_rows_by_persona(rows: list[tuple[int, dict]]) -> dict[int, list[tuple[int, dict]]]:
    grouped: dict[int, list[tuple[int, dict]]] = defaultdict(list)
    for row_idx, row in rows:
        grouped[int(row["persona_id"])].append((row_idx, row))
    return dict(grouped)


def search_persona_rows(
    persona_id: int,
    rows: list[tuple[int, dict]],
    *,
    frame: str,
    version: str,
    top_k: int,
    allow_empty_search: bool,
) -> list[float]:
    memory_user_id = user_id_for(version, persona_id)
    client = create_client(frame)
    durations: list[float] = []
    for row_idx, row_data in rows:
        tmp_path = _row_tmp_path(frame, version, row_idx)
        if tmp_path.exists():
            print(f"  Using existing PersonaMem search result for row {row_idx}")
            continue

        question = parse_user_query(row_data["user_query"])
        pref_type = row_data["pref_type"]
        topic = row_data["topic_query"]
        correct_answer = row_data["correct_answer"]
        all_options, golden_answer = build_options(
            correct_answer,
            row_data["incorrect_answers"],
            row_idx,
        )

        print(f"  Searching PersonaMem persona {persona_id}, row {row_idx}: {question[:100]}...")
        result = dispatch_search(frame, client, question, memory_user_id, top_k)
        context, duration_ms, reflect_answer, raw_context = unpack_search_result(result)
        durations.append(duration_ms)
        context = context or ""
        status = classify_search_status(
            context,
            reflect_answer,
            raw_context=raw_context,
        )
        if status == STATUS_SUCCESS_EMPTY and not allow_empty_search:
            raise RuntimeError(
                f"search returned no raw memories for PersonaMem persona {persona_id}, row {row_idx}"
            )

        result_key = result_key_for(version, row_idx)
        entry = build_search_entry(
            result_key=result_key,
            persona_id=persona_id,
            row_idx=row_idx,
            question=question,
            category=pref_type,
            all_options=all_options,
            topic=topic,
            golden_answer=golden_answer,
            context=context,
            duration_ms=duration_ms,
            status=status,
            reflect_answer=reflect_answer,
        )

        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json_dump({result_key: [entry]}, tmp_path, indent=4)

    return durations


def write_combined_results(
    frame: str,
    version: str,
    completed: set[str],
    rows_by_persona: dict[int, list[tuple[int, dict]]],
) -> None:
    combined: dict[str, list] = defaultdict(list)
    for persona_key in sorted(completed, key=lambda x: int(x)):
        persona_id = int(persona_key)
        for row_idx, _ in rows_by_persona.get(persona_id, []):
            path = _row_tmp_path(frame, version, row_idx)
            if not path.exists():
                continue
            with path.open() as f:
                data = json.load(f)
            for user_id, entries in data.items():
                combined[user_id].extend(entries)

    out = _combined_path(frame, version)
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(dict(combined), out, indent=4)


def write_search_status(
    frame: str,
    version: str,
    completed: set[str],
    rows_by_persona: dict[int, list[tuple[int, dict]]],
    *,
    allow_empty_search: bool,
    skip_failed_streaming: bool,
    failed_users: list[dict],
    skipped_records: list[dict],
) -> None:
    records = []
    for persona_key in sorted(completed, key=lambda x: int(x)):
        persona_id = int(persona_key)
        for row_idx, _ in rows_by_persona.get(persona_id, []):
            path = _row_tmp_path(frame, version, row_idx)
            if not path.exists():
                continue
            with path.open() as f:
                data = json.load(f)
            for entries in data.values():
                if isinstance(entries, list):
                    records.extend(entry for entry in entries if isinstance(entry, dict))

    atomic_json_dump(
        {
            "stage": "search",
            "mode": "personamem_streaming",
            "allow_empty_search": allow_empty_search,
            "skip_failed_streaming": skip_failed_streaming,
            "status_counts": status_counts(records),
            "failed_users": failed_users,
            "skipped_records": skipped_records,
        },
        _results_dir(frame, version) / f"{frame}_pm_search_status.json",
        indent=2,
    )


def load_stats(frame: str, version: str) -> dict:
    path = _stats_path(frame, version)
    if path.exists():
        with path.open() as f:
            return json.load(f)
    return {
        "mode": "personamem_streaming",
        "user_durations_ms": {},
        "add_call_durations_by_unit": {},
        "add_call_durations_ms": [],
        "search_call_durations_ms": [],
        "final_delete_statuses": {},
        "ingest_modes": {},
        "chunk_counts": {},
        "message_counts": {},
        "char_counts": {},
    }


def save_stats(frame: str, version: str, stats: dict) -> None:
    path = _stats_path(frame, version)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(stats, path, indent=2)


def process_persona(
    persona_id: int,
    chat_history_path: str,
    rows: list[tuple[int, dict]],
    *,
    frame: str,
    version: str,
    top_k: int,
    allow_empty_search: bool,
    ingest_mode: str,
    max_batch_chars: int,
    wait_after_ingest: float,
    completed: set[str],
    stats: dict,
    rows_by_persona: dict[int, list[tuple[int, dict]]],
    restart_unit: bool,
) -> None:
    user_id = user_id_for(version, persona_id)
    started = time.time()
    print("\n" + "=" * 80)
    print(f"PERSONAMEM STREAM persona {persona_id}: add({ingest_mode}), search, delete")
    print("=" * 80)

    with open(chat_history_path, encoding="utf-8") as fh:
        chat_history = json.load(fh)["chat_history"]

    client = create_client(frame)
    added_path = _added_chunks_path(frame, version, persona_id)
    added_chunks = load_added_chunks(added_path)
    supports_add_resume = ingest_mode in ("chunked", "per-chunk")
    should_start_fresh = restart_unit or not (supports_add_resume and added_chunks)

    if should_start_fresh:
        delete_user_data(frame, client, user_id, skip_errors=True)
        prepare_user_after_delete(frame, client, user_id)
        if added_path.exists():
            added_path.unlink()
        added_chunks = set()
        for row_idx, _ in rows:
            tmp_path = _row_tmp_path(frame, version, row_idx)
            if tmp_path.exists():
                tmp_path.unlink()
    else:
        print(f"Resuming PersonaMem persona {persona_id}: {len(added_chunks)} chunks recorded")

    if ingest_mode == "chunked":
        if frame != "graphiti":
            raise RuntimeError("chunked add requires a raw-chunk capable client")
        add_durations, meta = add_persona_graphiti_chunked(
            persona_id,
            chat_history,
            version,
            client,
            max_chars=max_batch_chars,
            added_chunks_path=added_path,
            added_chunks=added_chunks,
        )
    else:
        add_durations, meta = add_persona_per_chunk(
            persona_id,
            chat_history,
            frame,
            version,
            client,
            added_chunks_path=added_path,
            added_chunks=added_chunks,
        )

    if wait_after_ingest > 0:
        print(f"Waiting {wait_after_ingest}s after ingest")
        time.sleep(wait_after_ingest)

    search_durations = search_persona_rows(
        persona_id,
        rows,
        frame=frame,
        version=version,
        top_k=top_k,
        allow_empty_search=allow_empty_search,
    )

    final_delete_ok, final_delete_error = delete_user_data(
        frame,
        client,
        user_id,
        skip_errors=True,
    )
    final_delete_status = "ok" if final_delete_ok else "error_skipped"
    if final_delete_ok and added_path.exists():
        added_path.unlink()
    if not final_delete_ok:
        log_event(
            _events_path(frame, version),
            "final_delete_error_skipped",
            persona_id=persona_id,
            user_id=user_id,
            error=final_delete_error,
        )

    mark_completed(_completed_path(frame, version), completed, str(persona_id))
    elapsed_ms = round((time.time() - started) * 1000, 1)

    key = str(persona_id)
    stats.setdefault("user_durations_ms", {})[key] = elapsed_ms
    update_unit_duration_list(
        stats,
        key,
        add_durations,
        map_key="add_call_durations_by_unit",
        flat_key="add_call_durations_ms",
    )
    stats.setdefault("search_call_durations_ms", []).extend(round(v, 2) for v in search_durations)
    stats.setdefault("final_delete_statuses", {})[key] = final_delete_status
    stats.setdefault("ingest_modes", {})[key] = ingest_mode
    stats.setdefault("chunk_counts", {})[key] = meta["chunks"]
    stats.setdefault("message_counts", {})[key] = meta["messages"]
    stats.setdefault("char_counts", {})[key] = meta["chars"]
    if ingest_mode == "chunked":
        stats["max_batch_chars"] = max_batch_chars

    log_event(
        _events_path(frame, version),
        "completed",
        persona_id=persona_id,
        ingest_mode=ingest_mode,
        chunks=meta["chunks"],
        skipped_chunks=meta["skipped_chunks"],
        messages=meta["messages"],
        chars=meta["chars"],
        questions=len(rows),
        final_delete_status=final_delete_status,
    )
    write_combined_results(frame, version, completed, rows_by_persona)
    save_stats(frame, version, stats)


def mark_streaming_failure_skipped(
    persona_id: int,
    rows: list[tuple[int, dict]],
    frame: str,
    version: str,
    completed: set[str],
    exc: BaseException,
) -> dict:
    for row_idx, row_data in rows:
        tmp_path = _row_tmp_path(frame, version, row_idx)
        if tmp_path.exists():
            continue
        question = parse_user_query(row_data["user_query"])
        all_options, golden_answer = build_options(
            row_data["correct_answer"],
            row_data["incorrect_answers"],
            row_idx,
        )
        result_key = result_key_for(version, row_idx)
        entry = build_search_entry(
            result_key=result_key,
            persona_id=persona_id,
            row_idx=row_idx,
            question=question,
            category=row_data["pref_type"],
            all_options=all_options,
            topic=row_data["topic_query"],
            golden_answer=golden_answer,
            context="",
            duration_ms=0.0,
            status=STATUS_SKIPPED,
            error=error_payload("streaming", exc),
        )
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json_dump({result_key: [entry]}, tmp_path, indent=4)
    mark_completed(_completed_path(frame, version), completed, str(persona_id))
    return {
        "persona_id": persona_id,
        "user_id": user_id_for(version, persona_id),
        "error": error_payload("streaming", exc),
    }


def main() -> int:
    parser = argparse.ArgumentParser("PersonaMem v2 streaming add-search-delete")
    parser.add_argument("--lib", choices=SUPPORTED_LIBS, default=DEFAULT_LIB)
    parser.add_argument("--env", help="Dotenv file to load")
    parser.add_argument("--version", default="default")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--allow-empty-search",
        "--allow_empty_search",
        type=parse_bool,
        default=True,
        help="Allow successful searches with no raw memories. Default: 1.",
    )
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--end-idx", type=int)
    parser.add_argument("--wait-after-ingest", type=float, default=0.0)
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore streaming_completed.txt and process selected personas anyway.",
    )
    parser.add_argument(
        "--restart-unit",
        action="store_true",
        help="Delete each selected user and discard chunk/search checkpoints before reprocessing.",
    )
    parser.add_argument(
        "--skip-failed-streaming",
        action="store_true",
        help="Mark failed streaming units as skipped and continue instead of failing at the end.",
    )
    parser.add_argument(
        "--allow-missing-data",
        "--allow_missing_data",
        type=parse_bool,
        default=False,
        help="Allow missing PersonaMem chat histories and evaluate the available subset. Default: 0.",
    )
    args = parser.parse_args()

    if args.env:
        env_path = Path(args.env)
        if not env_path.is_file():
            env_path = Path.cwd() / args.env
        if not env_path.is_file():
            raise SystemExit(f"Env file not found: {args.env}")
        os.environ["OMNIMEMEVAL_ENV_FILE"] = str(env_path.resolve())

    load_env()
    configure_single_user_streaming(args.lib)

    max_batch_chars = resolve_max_batch_chars(args.lib)

    ingest_mode = "chunked" if args.lib == "graphiti" else "per-chunk"

    if ingest_mode == "chunked" and args.lib != "graphiti":
        raise SystemExit("chunked add is currently supported only for raw-chunk capable clients")

    results_dir = _results_dir(args.lib, args.version)
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "tmp").mkdir(parents=True, exist_ok=True)

    chat_history_index = build_chat_history_index(CHAT_HISTORY_DIR)
    benchmark_personas = load_benchmark_personas(BENCHMARK_CSV)
    missing_personas = sorted(set(benchmark_personas) - set(chat_history_index))
    if missing_personas and not args.allow_missing_data:
        raise SystemExit(
            f"Missing PersonaMem chat history files for {len(missing_personas)} "
            "benchmark personas. Use --allow-missing-data 1 to evaluate the available subset."
        )
    available_personas = [
        pid
        for pid in sorted(benchmark_personas)
        if pid in chat_history_index
    ]

    rows = load_benchmark_rows(BENCHMARK_CSV, chat_history_index)
    rows_by_persona = group_rows_by_persona(rows)

    total = len(available_personas)
    end_idx = args.end_idx if args.end_idx is not None else total - 1
    if args.start_idx < 0 or end_idx >= total or end_idx < args.start_idx:
        raise SystemExit(
            f"Invalid range start={args.start_idx}, end={end_idx}, total={total}"
        )

    completed_path = _completed_path(args.lib, args.version)
    existing_completed = load_completed(completed_path)
    completed = set() if args.no_resume else set(existing_completed)
    stats = load_stats(args.lib, args.version)

    print("\n" + "=" * 80)
    print("PERSONAMEM V2 STREAMING")
    print("=" * 80)
    print(f"lib={args.lib}")
    print(f"version={args.version}")
    print(f"range={args.start_idx}-{end_idx}")
    print(f"top_k={args.top_k}")
    print(f"allow_empty_search={args.allow_empty_search}")
    print(f"ingest_mode={ingest_mode}")
    if ingest_mode == "chunked":
        print(f"max_batch_chars={max_batch_chars}")
    print(f"wait_after_ingest={args.wait_after_ingest}")
    print(f"already_completed={len(completed)}")
    print("=" * 80)

    failed_users: list[dict] = []
    skipped_records: list[dict] = []
    for row_pos in range(args.start_idx, end_idx + 1):
        persona_id = available_personas[row_pos]
        restart_unit = args.restart_unit or (
            args.no_resume and str(persona_id) in existing_completed
        )
        if str(persona_id) in completed and not args.no_resume and not restart_unit:
            print(f"Skipping PersonaMem persona {persona_id}: already completed")
            continue
        try:
            process_persona(
                persona_id,
                chat_history_index[persona_id],
                rows_by_persona.get(persona_id, []),
                frame=args.lib,
                version=args.version,
                top_k=args.top_k,
                allow_empty_search=args.allow_empty_search,
                ingest_mode=ingest_mode,
                max_batch_chars=max_batch_chars,
                wait_after_ingest=args.wait_after_ingest,
                completed=completed,
                stats=stats,
                rows_by_persona=rows_by_persona,
                restart_unit=restart_unit,
            )
        except KeyboardInterrupt:
            print("\nInterrupted by user")
            raise
        except Exception as exc:
            log_event(
                _events_path(args.lib, args.version),
                "failed",
                persona_id=persona_id,
                error=str(exc),
            )
            print(f"ERROR PersonaMem persona {persona_id}: {type(exc).__name__}: {exc}")
            failure = {
                "persona_id": persona_id,
                "user_id": user_id_for(args.version, persona_id),
                "error": error_payload("streaming", exc),
            }
            if args.skip_failed_streaming:
                skipped_records.append(
                    mark_streaming_failure_skipped(
                        persona_id,
                        rows_by_persona.get(persona_id, []),
                        args.lib,
                        args.version,
                        completed,
                        exc,
                    )
                )
                write_combined_results(args.lib, args.version, completed, rows_by_persona)
                save_stats(args.lib, args.version, stats)
                continue
            failed_users.append(failure)
            continue

    write_combined_results(args.lib, args.version, completed, rows_by_persona)
    save_stats(args.lib, args.version, stats)
    write_search_status(
        args.lib,
        args.version,
        completed,
        rows_by_persona,
        allow_empty_search=args.allow_empty_search,
        skip_failed_streaming=args.skip_failed_streaming,
        failed_users=failed_users,
        skipped_records=skipped_records,
    )
    if failed_users:
        print(f"\nPersonaMem streaming failed for {len(failed_users)} persona(s)")
        return 1
    print("\nPersonaMem streaming complete")
    print(f"Combined search results: {_combined_path(args.lib, args.version)}")
    print(f"Completed personas: {len(completed)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
