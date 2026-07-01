# OmniMemEval

[中文版](./README_zh.md)

OmniMemEval is an evaluation framework for memory systems and memory-augmented agents. It provides two complementary evaluation tracks:

| Track | Evaluation Target | Benchmarks / Domains | Documentation |
|---|---|---|---|
| User Memory Evaluation | Memory backend APIs exposed through `add()`, `search()`, and `delete()` | LoCoMo, LongMemEval, BEAM, PersonaMem v2, HaluMem | [docs/user_memory/README.md](./docs/user_memory/README.md) |
| Agent Memory Evaluation | Agent runtimes equipped with memory plugins | AgentBench domains: reasoning, information retrieval, knowledge work, code implementation, software engineering | [docs/agent_memory/README.md](./docs/agent_memory/README.md) |

## Evaluation Tracks

### User Memory Evaluation

User Memory Evaluation measures the capability of memory backend systems through a standardized API adapter layer. The same benchmark pipeline can be run against mainstream memory products, self-hosted memory frameworks, or custom adapters by selecting a backend with `--lib`.

The pipeline covers ingestion, retrieval, answer generation, LLM-as-Judge evaluation, metric aggregation, and report generation. Results are written under `results/<benchmark>/<LIB>-<VERSION>/`.

Start here: [docs/user_memory/README.md](./docs/user_memory/README.md)

### Agent Memory Evaluation

Agent Memory Evaluation measures the task performance of an agent runtime after a memory plugin is installed. The current implementation is based on AgentBench and evaluates OpenClaw across five task domains. The memory-plugin protocol includes memory cleanup, training, memory settling, backup, restore, and test execution.

Results are written under `results/agentbench/`.

Start here: [docs/agent_memory/README.md](./docs/agent_memory/README.md)

## Installation

User Memory Evaluation and Agent Memory Evaluation use separate environments. For User Memory Evaluation, create the base Python environment from the repository root:

```bash
conda create -n omnimemeval python=3.12 -y
conda activate omnimemeval
pip install -r requirements.txt
```

Agent Memory Evaluation recommends a separate `agentmem` environment because AgentBench domains require OpenClaw and additional domain dependencies. See [docs/agent_memory/README.md](./docs/agent_memory/README.md).

Each evaluation track may require additional dependencies:

- User Memory Evaluation: memory backend credentials, ANSWER/EVAL LLM credentials, and benchmark-specific data preparation. See [docs/user_memory/README.md](./docs/user_memory/README.md).
- Agent Memory Evaluation: OpenClaw CLI, AgentBench data, and domain-specific dependencies such as Docker, LiveCodeBench, and BrowseComp-Plus indexing. See [docs/agent_memory/README.md](./docs/agent_memory/README.md).

## Quick Start

### User Memory Evaluation

```bash
cp env_examples/.env.memos .env.memos
python data/locomo/prepare_locomo.py
./scripts/run_locomo_eval.sh --lib memos --env .env.memos
```

### Agent Memory Evaluation

```bash
cp env_examples/.env.agent .env.agent
mkdir -p data/agentbench
huggingface-cli download EverMind-AI/EvoAgentBench \
  --repo-type dataset \
  --local-dir ./data/agentbench
./scripts/run_agent_eval.sh \
  --agent openclaw \
  --domain reasoning \
  --protocol test_only \
  --version smoke_agentbench \
  --trials 1 \
  --parallel 1
```

## Results

- User Memory public result snapshot: [docs/benchmark-results.md](./docs/benchmark-results.md)
- AgentBench flow validation: [docs/agent_memory/eval_res.md](./docs/agent_memory/eval_res.md)

## Repository Layout

```text
configs/
  agentbench/                 # Agent Memory Evaluation configs
data/                         # User Memory benchmark data preparation
docs/
  user_memory/README.md       # User Memory Evaluation guide
  agent_memory/README.md      # Agent Memory Evaluation guide
  benchmark-results.md        # User Memory public result snapshot
env_examples/                 # Environment templates for memory backends and agent evaluation
scripts/
  agentbench/                 # AgentBench runner implementation
  client_factory/             # User Memory backend adapters
  locomo/ longmemeval/ beam/  # User Memory benchmark pipelines
  personamem_v2/ halumem/
results/                      # Evaluation outputs
```

## License

See [LICENSE](./LICENSE). Third-party benchmark data keeps its upstream license; the OmniMemEval code license does not relicense external datasets. See [THIRD_PARTY_NOTICES.md](./THIRD_PARTY_NOTICES.md).
