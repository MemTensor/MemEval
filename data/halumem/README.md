# HaluMem Dataset

HaluMem is the first benchmark specifically designed for evaluating **operation-level hallucinations in memory systems**. It decomposes memory systems into three core operations — memory extraction, memory update, and memory QA — and introduces distractor content to test robustness under noisy conditions.

- Paper: [HaluMem: A Comprehensive Benchmark for Evaluating Hallucinations in Memory Systems](https://arxiv.org/abs/2511.03506)
- Repository: https://github.com/MemTensor/HaluMem
- Data source: Hugging Face [`IAAR-Shanghai/HaluMem`](https://huggingface.co/datasets/IAAR-Shanghai/HaluMem)
- License: CC BY-NC-ND 4.0

## Files

| File | Size | Description |
|------|------|-------------|
| `HaluMem-Medium.jsonl` | ~32.0 MB | Medium version: ~70 sessions / ~160K tokens per user |
| `HaluMem-Long.jsonl` | ~101.6 MB | Long version: ~120 sessions / ~1M tokens per user (with extensive distractors) |

Only `README.md` and `prepare_halumem.py` are intended to be
version-controlled by OmniMemEval. Downloaded JSONL files are generated artifacts
and should remain local.

## Dataset Statistics

| Metric | Medium | Long |
|--------|--------|------|
| Users | 20 | 20 |
| Total sessions | 1,387 | 2,417 |
| Avg sessions/user | 69.3 | 120.8 |
| Total dialogue turns | 30,073 | 53,516 |
| Total tokens | 3,198,219 | 20,145,293 |
| Avg tokens/user | ~160K | ~1M |
| Memory points | 14,948 | 14,948 |
| QA pairs | 3,467 | 3,467 |

Both versions share the same users, memory points, and QA questions. The Long version extends per-user context from ~160K to ~1M tokens by adding distractor content (factual QA, math problems, etc.).

## Memory Type Distribution

| Type | Count | Share | Description |
|------|-------|-------|-------------|
| Persona Memory | 9,116 | 61.0% | Personal information (name, occupation, preferences, etc.) |
| Event Memory | 4,550 | 30.4% | Events and experiences |
| Relationship Memory | 1,282 | 8.6% | Interpersonal relationships |

## QA Question Types

| Type | Count | Description |
|------|-------|-------------|
| Memory Boundary | 828 | Identify questions beyond known memory scope (answer "unknown") |
| Memory Conflict | 769 | Handle contradictory memory information |
| Basic Fact Recall | 746 | Extract specific facts from memory |
| Generalization & Application | 746 | Reason and generalize based on memory |
| Multi-hop Inference | 198 | Reason across multiple memory entries |
| Dynamic Update | 180 | Identify memory updates and answer with the latest version |

## Data Structure

JSON Lines format, one user object per line.

**User object fields**: `uuid`, `persona_info`, `sessions`, `total_dialogue_token_length`, `total_question_count`, `token_cost`

**Session fields**: `start_time`, `end_time`, `memory_points_count`, `memory_points`, `dialogue_turn_num`, `dialogue` (list of `{role, content}`), `dialogue_token_length`, `questions`, `question_count`

**Memory point fields**: `index`, `memory_content`, `memory_type`, `is_update`, `original_memories`, `timestamp`, `importance`, `memory_source`

**Question fields**: `question`, `answer`, `evidence`, `difficulty`, `question_type`

## Data Preparation

The JSONL files are downloaded directly from Hugging Face
[`IAAR-Shanghai/HaluMem`](https://huggingface.co/datasets/IAAR-Shanghai/HaluMem).
The default download is Medium, matching OmniMemEval's runner default.

```bash
python data/halumem/prepare_halumem.py                        # download medium (~32 MB, used by default)
python data/halumem/prepare_halumem.py --variant medium long  # download all variants
python data/halumem/prepare_halumem.py --force                # overwrite existing files
```

The script validates each downloaded JSONL file before replacing the local copy.

## License And Redistribution

HaluMem is released under
[Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)](https://creativecommons.org/licenses/by-nc-nd/4.0/).
Do not redistribute modified copies of the dataset, and keep use within the
license terms. The OmniMemEval code license does not override the upstream HaluMem
dataset license.

## Evaluation

- **Metric**: LLM-as-a-Judge Accuracy / F1
- **Pipeline**: Ingest sessions → Search memories → Generate answers → Evaluate
- **Three core operations evaluated**: Memory Extraction (recall), Memory Update (accuracy), Memory QA (accuracy/F1)

## References

- Paper: [HaluMem: A Comprehensive Benchmark for Evaluating Hallucinations in Memory Systems](https://arxiv.org/abs/2511.03506)
- Repository: https://github.com/MemTensor/HaluMem
- Hugging Face: https://huggingface.co/datasets/IAAR-Shanghai/HaluMem
