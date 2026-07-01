"""LLM Judge for BrowseComp-Plus.

Adapted from the BrowseComp-Plus evaluation script.
Uses an OpenAI-compatible API (sglang / vllm) to judge answer correctness.
"""

import logging
import re

log = logging.getLogger("agentbench")


def _error_text(exc: Exception) -> str:
    """Best-effort extraction of useful OpenAI error details."""
    response = getattr(exc, "response", None)
    if response is not None:
        try:
            return response.text
        except Exception:
            pass
    return str(exc)


def _completion_kwargs(
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    disable_thinking: bool,
    token_param: str,
    include_temperature: bool,
) -> dict:
    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        token_param: max_tokens,
    }
    if include_temperature:
        kwargs["temperature"] = temperature
    if disable_thinking:
        kwargs["extra_body"] = {
            "chat_template_kwargs": {"enable_thinking": False}
        }
    return kwargs


def _call_chat_completion_compat(
    client,
    *,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    disable_thinking: bool,
):
    """Call chat.completions with fallbacks for OpenAI-compatible endpoints."""
    token_param = "max_tokens"
    include_temperature = True
    last_err = None

    for _ in range(5):
        kwargs = _completion_kwargs(
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            disable_thinking=disable_thinking,
            token_param=token_param,
            include_temperature=include_temperature,
        )
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as exc:
            last_err = exc
            msg = _error_text(exc)
            msg_lower = msg.lower()

            if disable_thinking and "chat_template_kwargs" in msg:
                disable_thinking = False
                log.warning(
                    "Judge endpoint rejected chat_template_kwargs; retrying without it"
                )
                continue

            if token_param == "max_tokens" and (
                "max_tokens" in msg_lower
                and (
                    "max_completion_tokens" in msg_lower
                    or "unsupported" in msg_lower
                    or "not support" in msg_lower
                    or "unrecognized" in msg_lower
                    or "unknown" in msg_lower
                )
            ):
                token_param = "max_completion_tokens"
                log.warning(
                    "Judge endpoint rejected max_tokens; retrying with max_completion_tokens"
                )
                continue

            if include_temperature and (
                "temperature" in msg_lower
                and (
                    "unsupported" in msg_lower
                    or "not support" in msg_lower
                    or "only the default" in msg_lower
                    or "must be" in msg_lower
                )
            ):
                include_temperature = False
                log.warning(
                    "Judge endpoint rejected temperature; retrying without it"
                )
                continue

            raise

    raise last_err


GRADER_TEMPLATE = """
Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

[correct_answer]: {correct_answer}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response].

[correct_answer]: Repeat the [correct_answer] given above.

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], in the context of this [question]. You should judge whether the extracted_final_answer is semantically equivalent to [correct_answer], allowing the extracted_final_answer to be string variations of [correct_answer]. You should also allow the extracted_final_answer to be more precise or verbose than [correct_answer], as long as its additional details are correct. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers are semantically equivalent.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.


confidence: The extracted confidence score between 0|%| and 100|%| from [response]. Put 100 if there is no confidence score available.
""".strip()


def create_judge_prompt(question: str, response: str, correct_answer: str) -> str:
    return GRADER_TEMPLATE.format(
        question=question, response=response, correct_answer=correct_answer,
    )


def parse_judge_response(judge_response: str) -> dict:
    """Parse the structured judge response into a result dict."""
    result = {
        "extracted_final_answer": None,
        "reasoning": None,
        "correct": None,
        "confidence": None,
        "parse_error": False,
    }
    if not judge_response:
        result["parse_error"] = True
        return result

    # Extract extracted_final_answer
    for pattern in [
        r"\*\*extracted_final_answer:\*\*\s*(.*?)(?=\n|$)",
        r"\*\*extracted_final_answer\*\*:\s*(.*?)(?=\n|$)",
        r"extracted_final_answer:\s*(.*?)(?=\n|$)",
    ]:
        m = re.search(pattern, judge_response, re.IGNORECASE | re.DOTALL)
        if m:
            result["extracted_final_answer"] = m.group(1).strip()
            break

    # Extract correct (yes/no)
    for pattern in [
        r"\*\*correct:\*\*\s*(yes|no)",
        r"\*\*correct\*\*:\s*(yes|no)",
        r"correct:\s*(yes|no)",
    ]:
        m = re.search(pattern, judge_response, re.IGNORECASE)
        if m:
            result["correct"] = m.group(1).lower() == "yes"
            break

    # Extract confidence
    for pattern in [
        r"\*\*confidence:\*\*\s*(\d+(?:\.\d+)?)\s*%?",
        r"\*\*confidence\*\*:\s*(\d+(?:\.\d+)?)\s*%?",
        r"confidence:\s*(\d+(?:\.\d+)?)\s*%?",
    ]:
        m = re.search(pattern, judge_response, re.IGNORECASE)
        if m:
            result["confidence"] = min(float(m.group(1)), 100)
            break

    if result["correct"] is None:
        result["parse_error"] = True

    return result


def call_judge(question: str, response: str, correct_answer: str,
               model: str, api_base: str, api_key: str = "EMPTY",
               max_tokens: int = 4096, temperature: float = 0.7,
               disable_thinking: bool = True) -> dict:
    """Call an OpenAI-compatible LLM judge and return parsed result."""
    import os
    import openai

    # Support ${ENV_VAR} syntax in api_key
    if api_key.startswith("${") and api_key.endswith("}"):
        api_key = os.environ.get(api_key[2:-1], "")

    client = openai.OpenAI(api_key=api_key, base_url=api_base)
    prompt = create_judge_prompt(question, response, correct_answer)

    try:
        resp = _call_chat_completion_compat(
            client,
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            disable_thinking=disable_thinking,
        )
        judge_text = resp.choices[0].message.content or ""
    except Exception as e:
        log.error(f"Judge API error: {_error_text(e)}")
        return {"parse_error": True, "error": _error_text(e)}

    result = parse_judge_response(judge_text)
    result["judge_response"] = judge_text
    return result
