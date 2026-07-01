from __future__ import annotations

import json


FEEDBACK_PREFIX = "Verifier feedback for the previous attempt."


def build_feedback_prompt(task_name: str, verifier_result: dict) -> str:
    reward = verifier_result.get("reward", 0.0)
    feedback = verifier_result.get("feedback")

    lines = [
        FEEDBACK_PREFIX,
        "The benchmark verifier checked your previous answer and returned the result below.",
        f"Related benchmark case: {task_name}",
        f"Verifier reward: {reward}",
    ]
    if feedback:
        lines.extend(["", "Verifier feedback:", str(feedback)])

    lines.extend([
        "",
        "Verifier result JSON:",
        json.dumps(verifier_result, ensure_ascii=False, indent=2),
        "",
        "Please briefly reflect on what you would keep and what you would improve next time.",
    ])
    return "\n".join(lines)
