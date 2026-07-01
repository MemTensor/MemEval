# PersonaMem v2 Dataset

PersonaMem v2 is a persona-centric personalized memory benchmark that evaluates whether LLMs can adapt responses based on user traits, preferences, and interaction history across multi-session conversations. Published by University of Pennsylvania.

- Paper: [PersonaMem-v2: Towards Personalized Intelligence via Learning Implicit User Personas and Agentic Memory](https://arxiv.org/abs/2512.06688)
- Repository: https://github.com/bowen-upenn/PersonaMem-v2
- Data source: Hugging Face [`bowen-upenn/PersonaMem-v2`](https://huggingface.co/datasets/bowen-upenn/PersonaMem-v2)
- License: MIT

## Files

| File / Directory | Size | Description |
|------------------|------|-------------|
| `benchmark/text/benchmark.csv` | ~40.5 MB | Benchmark set: 5,000 user queries across 200 personas |
| `benchmark/text/train.csv` | ~149.8 MB | Training set: 18,549 queries (no persona overlap with benchmark) |
| `benchmark/text/val.csv` | ~16.7 MB | Validation set: 2,061 queries (no persona overlap with benchmark) |
| `column_descriptions.md` | ~3.7 KB | Detailed field descriptions for CSV columns |
| `combined_irrelevant_data.json` | ~1.0 MB | 1,545 irrelevant conversation snippets (math/coding) for context padding |
| `data/chat_history_32k/` | — | 32K-token context chat histories (1 per persona) |
| `data/chat_history_128k/` | — | 128K-token context chat histories (1 per persona) |
| `data/raw_data/` | — | Raw persona JSON files with full profiles and preferences |

Only `README.md` and `prepare_personamem.py` are intended to be
version-controlled by OmniMemEval. Downloaded CSV/JSON files are generated artifacts
and should remain local.

> **Note**: The complete dataset includes 1,000 personas. If additional data is needed, download from Hugging Face.

## Dataset Statistics

| Metric | Value |
|--------|-------|
| Benchmark queries | 5,000 |
| Training queries | 18,549 |
| Validation queries | 2,061 |
| Unique personas | 1,000 |
| 32K context avg tokens | ~31,842 |
| 128K context avg tokens | ~104,178 |
| Preference types | 7 |
| Topic categories | 20+ |

## Key CSV Fields

| Field | Description |
|-------|-------------|
| `persona_id` | Unique persona identifier |
| `short_persona` / `expanded_persona` | Brief / detailed persona description |
| `user_query` | User query (JSON format) |
| `correct_answer` | Personalized correct response |
| `incorrect_answers` | Three plausible but incorrect alternatives |
| `preference` | User preference being tested |
| `pref_type` | Preference category |
| `updated` | Whether the preference has been updated |
| `sensitive_info` | Whether sensitive information is involved |

See `column_descriptions.md` for the complete field reference.

## Preference Type Distribution

| Preference Type | Count |
|-----------------|-------|
| `ask_to_forget` | 1,048 |
| `neutral_preferences` | 858 |
| `anti_stereotypical_pref` | 855 |
| `therapy_background` | 627 |
| `health_and_medical_conditions` | 568 |
| `stereotypical_pref` | 533 |
| `sensitive_info` | 511 |

## Data Preparation

The dataset files are downloaded from Hugging Face
[`bowen-upenn/PersonaMem-v2`](https://huggingface.co/datasets/bowen-upenn/PersonaMem-v2).
If `huggingface.co` is unreachable, set `HF_ENDPOINT` (for example
`https://hf-mirror.com`) and optionally `HF_HUB_DOWNLOAD_TIMEOUT=300`.

The default command downloads the files required by OmniMemEval: benchmark CSV plus
the 32K chat histories referenced by that CSV.

```bash
pip install huggingface_hub
python data/personamem_v2/prepare_personamem.py                           # download eval-required files only
python data/personamem_v2/prepare_personamem.py --include 128k            # also download 128K chat histories
python data/personamem_v2/prepare_personamem.py --include all-csv raw-data # download extra data
python data/personamem_v2/prepare_personamem.py --verify-chat-32k         # validate existing 32K chat histories
```

The script validates the benchmark CSV schema and checks that downloaded 32K
chat history files parse as JSON.

## License And Redistribution

`bowen-upenn/PersonaMem-v2` is published on Hugging Face with the MIT license.
The OmniMemEval code license does not override the upstream PersonaMem v2 dataset
license.

## Evaluation

- **Metric**: Accuracy over multiple-choice personalized responses
- **Pipeline**: Ingest chat history → Search relevant context → Generate personalized response → Extract selected option → Compute accuracy
- **Two context scales**: 32K tokens and 128K tokens
- **Core capabilities tested**: Preference identification, personalized response generation, sensitive information handling, preference forgetting, preference update tracking

## References

- Paper: [PersonaMem-v2: Towards Personalized Intelligence via Learning Implicit User Personas and Agentic Memory](https://arxiv.org/abs/2512.06688)
- Repository: https://github.com/bowen-upenn/PersonaMem-v2
- Hugging Face: https://huggingface.co/datasets/bowen-upenn/PersonaMem-v2
