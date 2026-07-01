import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))

from agentbench.report import generate_all_domains_report
from agentbench.run_agent_eval import build_run_dir_name


def _write_summary(root: Path, domain: str, version: str, **overrides):
    phase_dir = root / f"openclaw-{version}-{domain}" / "test"
    phase_dir.mkdir(parents=True)
    summary = {
        "tasks": 2,
        "total_trials": 2,
        "trials_per_task": 1,
        "pass_at": 1,
        "pass@1": 0.5,
        "avg_pass_rate": 0.5,
        "avg_elapsed_sec": 10.0,
        "avg_tokens": 100,
        "averages": {"avg_reward": 0.5},
        "failure_counts": {"failed": 1, "resolved": 1},
        "domain_metrics": {"score": 0.25},
    }
    summary.update(overrides)
    (phase_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return phase_dir


def test_generate_all_domains_report_writes_json_and_markdown(tmp_path):
    results_root = tmp_path / "results"
    report_dir = tmp_path / "reports"
    version = "openclaw_all5_test_20260701_010203"
    domains = ["reasoning", "knowledge_work"]

    _write_summary(results_root, "reasoning", version, **{"pass@1": 1.0, "avg_tokens": 50})
    _write_summary(
        results_root,
        "knowledge_work",
        version,
        **{
            "pass@1": 0.0,
            "avg_tokens": 150,
            "infra_excluded": {"pass@1": {"mean": 0.75, "stderr": 0.0}},
        },
    )

    report = generate_all_domains_report(
        results_root=results_root,
        report_dir=report_dir,
        agent="openclaw",
        version=version,
        domains=domains,
        trials=1,
        parallel=4,
    )

    assert (report_dir / f"{version}_report.json").exists()
    assert (report_dir / f"{version}_report.md").exists()
    assert report["agent"] == "openclaw"
    assert report["version"] == version
    assert report["runs"] == 1
    assert report["parallel"] == 4
    assert report["domains"]["reasoning"]["average_pass_at_1"] == 1.0
    assert report["domains"]["knowledge_work"]["average_pass_at_1"] == 0.0
    assert report["domains"]["knowledge_work"]["average_infra_excluded_pass_at_1"] == 0.75
    assert report["overall"]["average_pass_at_1"] == 0.5
    assert report["domains"]["reasoning"]["runs"][0]["summary"].endswith("summary.json")
    assert report["domains"]["reasoning"]["runs"][0]["avg_tokens"] == 50


def test_generate_all_domains_report_supports_legacy_domain_first_dirs(tmp_path):
    results_root = tmp_path / "results"
    report_dir = tmp_path / "reports"
    version = "all5_test_20260701_010203"
    phase_dir = results_root / f"openclaw-reasoning-{version}" / "test"
    phase_dir.mkdir(parents=True)
    (phase_dir / "summary.json").write_text(
        json.dumps({"tasks": 1, "pass@1": 0.25, "avg_elapsed_sec": 3.0}),
        encoding="utf-8",
    )

    report = generate_all_domains_report(
        results_root=results_root,
        report_dir=report_dir,
        agent="openclaw",
        version=version,
        domains=["reasoning"],
        trials=1,
        parallel=1,
    )

    run = report["domains"]["reasoning"]["runs"][0]
    assert run["pass_at_1"] == 0.25
    assert run["job"] == f"openclaw-reasoning-{version}"


def test_generate_all_domains_report_fails_when_summary_missing(tmp_path):
    try:
        generate_all_domains_report(
            results_root=tmp_path / "results",
            report_dir=tmp_path / "reports",
            agent="openclaw",
            version="missing_run",
            domains=["reasoning"],
            trials=1,
            parallel=1,
        )
    except FileNotFoundError as exc:
        assert "Missing AgentBench summary" in str(exc)
    else:
        raise AssertionError("missing summary should fail")


def test_build_run_dir_name_groups_by_version_before_domain():
    assert (
        build_run_dir_name(profile_name="openclaw", domain="reasoning", version="all5_test_20260701_010203")
        == "openclaw-all5_test_20260701_010203-reasoning"
    )
