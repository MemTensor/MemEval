# AgentBench 评测结果记录

本文记录 OmniMemEval AgentBench 迁移后的流程验证结果。目的主要是确认五个域的 train/test、记忆生命周期、feedback、structured feedback 和 OpenClaw adapter 行为是否正确，不作为正式排行榜结论。

## 验证环境

- 日期：2026-07-01
- 项目目录：`/root/gyh/OmniMemEval_dev/OmniMemEval`
- Agent：`openclaw`
- Agent 配置：`configs/agentbench/agents/openclaw.yaml`
- 记忆插件：`memos`
- 记忆插件配置：`configs/agentbench/memory_plugins/memos.yaml`
- 协议：`memory_train_backup_test`
- 每域样本：1 条 train + 1 条 test
- `test_runs`: 1
- `trials`: 1
- `parallel`: 1
- train feedback：启用
- structured feedback：启用
- test 前恢复记忆：启用

## 最终验证结果

最终有效结果由两批组成：

- 四个非 SWE 域：`memos_5domain_toolfix_20260701_192800`
- SWE 域：`memos_swe_trajectoryfix_20260701_202824`

| 域 | 结果目录 | Train pass@1 | Test pass@1 | Train reward | Test reward | Structured feedback | 记忆生命周期 |
|---|---|---:|---:|---:|---:|---|---|
| reasoning | `openclaw-memos-memos_5domain_toolfix_20260701_192800-reasoning` | 1.0 | 0.0 | 1.0 | 0.0 | submitted, same_episode=true, turns=2 | clear/backup/restore 全部成功 |
| information_retrieval | `openclaw-memos-memos_5domain_toolfix_20260701_192800-information_retrieval` | 1.0 | 0.0 | 1.0 | 0.0 | submitted, same_episode=true, turns=2 | clear/backup/restore 全部成功 |
| knowledge_work | `openclaw-memos-memos_5domain_toolfix_20260701_192800-knowledge_work` | 1.0 | 0.0 | 0.65 | 0.0 | submitted, same_episode=true, turns=2 | clear/backup/restore 全部成功 |
| code_implementation | `openclaw-memos-memos_5domain_toolfix_20260701_192800-code_implementation` | 1.0 | 1.0 | 1.0 | 1.0 | submitted, same_episode=true, turns=2 | clear/backup/restore 全部成功 |
| software_engineering | `openclaw-memos-memos_swe_trajectoryfix_20260701_202824-software_engineering` | 0.0 | 1.0 | 0.0 | 1.0 | submitted, same_episode=true, turns=2 | clear/backup/restore 全部成功 |

## 关键检查项

| 检查项 | 结果 |
|---|---|
| 五个域均能完成 train + backup + restore + test | 通过 |
| train 后 feedback 使用同一个 OpenClaw session id | 通过 |
| structured feedback 能提交到同一 episode | 通过 |
| 每个 test run 前恢复对应 domain backup | 通过 |
| 结果目录落在 `results/agentbench` | 通过 |
| OpenClaw `toolUse` 中间状态不会被误判为最终回答 | 通过 |
| OpenClaw session 已写最终回答但 CLI 不退出时能恢复 | 通过 |
| OpenClaw trajectory 已 error/timeout 但 CLI 不退出时能结束 | 通过 |

## 运行中修复的问题

### 1. `toolUse` 被误当成最终回答

早期恢复逻辑只要看到 assistant 有 `stopReason` 就认为本轮已完成。工具型任务中 `stopReason=toolUse` 只是中间状态，导致 IR/KW/code/SWE 被截断在“我要搜索/读文件/写代码”的初始阶段。

修复：

- OpenClaw adapter 只把非 `toolUse` 的 assistant stop reason 当作终止型完成信号。
- feedback 与 train 共用 session 时，只接受本次调用之后新增的终止型 assistant 消息。

修复后效果：

- IR train 从截断失败变为完整检索并通过。
- KW train 生成 Excel 交付物并通过。
- code train/test 均通过。
- SWE train 能产生 patch 并运行 verifier。

### 2. trajectory 已结束但 CLI 不退出

SWE test 曾复现 OpenClaw trajectory 已写入：

```text
session.ended status=error idleTimedOut=true
```

但 CLI 进程仍然存活，runner 会继续等到 `agent_timeout`。修复后 adapter 监听 `.trajectory.jsonl`，如果 OpenClaw 已记录 error/timeout，会终止 CLI 并返回 `completion_status=timeout`，避免长时间空等。

注意：该场景下 SWE verifier 仍可使用 agent 已产生的 patch 继续验证，所以最终 test reward 可以为 1.0。

## 单测

修复后相关单测：

```bash
PYTHONPATH=scripts /root/miniconda3/envs/agentmem/bin/python -m pytest -q \
  scripts/tests/unit/test_agentbench_openclaw_config.py \
  scripts/tests/unit/test_agentbench_memos_feedback.py \
  scripts/tests/unit/test_agentbench_session.py \
  scripts/tests/unit/test_agentbench_reasoning.py
```

结果：

```text
23 passed in 1.90s
```

## 复现命令

五域运行命令：

```bash
PYTHON=/root/miniconda3/envs/agentmem/bin/python \
./scripts/run_agentbench_memory_train_backup_test.sh \
  --memory-plugin memos \
  --version memos_5domain_toolfix_20260701_192800 \
  --test-runs 1 \
  --trials 1 \
  --parallel 1
```

SWE 单域补充验证命令：

```bash
PYTHON=/root/miniconda3/envs/agentmem/bin/python \
./scripts/run_agent_eval.sh \
  --agent openclaw \
  --domain software_engineering \
  --protocol memory_train_backup_test \
  --memory-plugin memos \
  --version memos_swe_trajectoryfix_20260701_202824 \
  --test-runs 1 \
  --trials 1 \
  --parallel 1 \
  --max-retries 0
```

## 说明

- 本结果是每个域各 1 条样本的流程验证，不代表完整 benchmark 平均性能。
- reasoning、IR、KW 的 test reward 为 0 不代表流程失败，只说明对应单条 test 样本未通过 verifier。
- 外部 LLM judge 和 embedding 服务的延迟会显著影响 wall time，尤其是 IR judge。
