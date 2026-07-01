from __future__ import annotations

from .code_implementation.livecode import LiveCodeBenchAdapter
from .information_retrieval.browsecomp_plus import BrowseCompPlusAdapter
from .knowledge_work.gdpval import GDPValAdapter
from .reasoning.omnimath import ReasoningDomain
from .software_engineering.swebench import SWEBenchAdapter


DOMAIN_REGISTRY = {
    "code_implementation": LiveCodeBenchAdapter,
    "information_retrieval": BrowseCompPlusAdapter,
    "knowledge_work": GDPValAdapter,
    "reasoning": ReasoningDomain,
    "software_engineering": SWEBenchAdapter,
}


def create_domain(name: str, config: dict):
    try:
        cls = DOMAIN_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"Unknown domain {name!r}. Supported: {sorted(DOMAIN_REGISTRY)}") from exc
    return cls(config)
