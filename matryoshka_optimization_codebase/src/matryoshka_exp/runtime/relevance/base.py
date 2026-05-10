from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class RelevanceArtifacts:
    pair_relevance: pd.DataFrame
    report: dict


RELEVANCE_COLUMNS = [
    "qid",
    "docno",
    "relevance_estimated",
    "relevance_final",
    "confidence",
    "uncertainty",
    "source_mode",
    "is_qrel_overridden",
]
