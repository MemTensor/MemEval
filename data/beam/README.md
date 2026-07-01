# BEAM Dataset

BEAM (Beyond a Million Tokens) is a long-term memory benchmark designed for **extreme-length scenarios**, covering four scales from 128K to 10M tokens. Published by Université de Montréal et al. at ICLR 2026.

- Paper: [Beyond a Million Tokens: Benchmarking and Enhancing Long-Term Memory in LLMs](https://arxiv.org/abs/2510.27246) (ICLR 2026)
- Repository: https://github.com/mohammadtavakoli78/BEAM
- Data source: Hugging Face [`Mohammadta/BEAM`](https://huggingface.co/datasets/Mohammadta/BEAM) and [`Mohammadta/BEAM-10M`](https://huggingface.co/datasets/Mohammadta/BEAM-10M)
- License: CC BY-SA 4.0

## Files

| File | Size | Conversations | Questions | Description |
|------|------|---------------|-----------|-------------|
| `beam_100k.json` | ~14.1 MB | 20 | 400 | 128K tokens scale |
| `beam_500k.json` | ~85.9 MB | 35 | 700 | 500K tokens scale |
| `beam_1m.json` | ~172.3 MB | 35 | 700 | 1M tokens scale |
| `beam_10m_10m.json` | ~979.5 MB | 10 | 200 | 10M tokens scale |

Only `README.md` and `prepare_beam.py` are intended to be version-controlled by
OmniMemEval. Downloaded JSONL files are generated artifacts and should remain local.

## Dataset Statistics

| Metric | Value |
|--------|-------|
| Total conversations | 100 |
| Total questions | 2,000 |
| Scale levels | 4 (128K / 500K / 1M / 10M tokens) |
| Memory capability dimensions | 10 |
| Questions per conversation | 20 (2 per dimension) |

## 10 Memory Capability Dimensions

| Dimension | Questions/Conversation | Description |
|-----------|----------------------|-------------|
| `abstention` | 2 | Identify questions unanswerable from conversation |
| `contradiction_resolution` | 2 | Detect and resolve contradictory information |
| `event_ordering` | 2 | Correctly order events chronologically |
| `information_extraction` | 2 | Extract specific facts from conversation |
| `instruction_following` | 2 | Follow instructions given in conversation |
| `knowledge_update` | 2 | Track information updates over time |
| `multi_session_reasoning` | 2 | Integrate information across sessions |
| `preference_following` | 2 | Identify and follow user preferences |
| `summarization` | 2 | Accurately summarize conversation content |
| `temporal_reasoning` | 2 | Reason about temporal relationships |

## Data Structure

JSON Lines format. Each conversation object contains:

| Field | Type | Description |
|-------|------|-------------|
| `conversation_id` | string | Unique conversation identifier |
| `conversation_seed` | dict | Seed info with `category` (Coding/General/Math), `subtopics` |
| `narratives` | string | Conversation narrative outline |
| `user_profile` | dict | User profile information |
| `conversation_plan` | string | Conversation plan |
| `user_questions` | list | User questions with `messages` and `time_anchor` |
| `chat` | list | Full dialogue with `content`, `role`, `time_anchor`, `question_type` per turn |
| `probing_questions` | string | 20 test questions (JSON string) organized by 10 dimensions |

## Data Preparation

The JSON Lines files are converted from the original Hugging Face Parquet
datasets. The default download matches the OmniMemEval runner default
(`--scale 100k`) so that a new developer does not accidentally fetch the full
1.3 GB converted dataset.

```bash
pip install datasets
python data/beam/prepare_beam.py                   # download 100k (~14 MB)
python data/beam/prepare_beam.py --scale 100k      # download a single scale
python data/beam/prepare_beam.py --scale 100k 500k # download selected scales
python data/beam/prepare_beam.py --scale all       # download all scales
python data/beam/prepare_beam.py --force           # overwrite existing files
```

> Note: The 10M scale file is ~979 MB after JSONL conversion. Use `--scale`
> to selectively download smaller scales first.

## License And Redistribution

The BEAM datasets are released under
[Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/).
Redistribution or adapted copies must preserve attribution and comply with the
ShareAlike terms. The OmniMemEval code license does not override the upstream BEAM
dataset license.

## Evaluation

- **Metric**: Nugget Score (LLM-as-a-Judge), scored per atomic nugget (1.0 / 0.5 / 0.0)
- **Pipeline**: Ingest conversations → Search memories → Answer probing questions → LLM judge scores against rubric
- **Scale-stratified reporting**: Separate scores for 128K / 500K / 1M / 10M

## References

- Paper: [Beyond a Million Tokens: Benchmarking and Enhancing Long-Term Memory in LLMs](https://arxiv.org/abs/2510.27246)
- Repository: https://github.com/mohammadtavakoli78/BEAM
- Hugging Face: https://huggingface.co/datasets/Mohammadta/BEAM / https://huggingface.co/datasets/Mohammadta/BEAM-10M
