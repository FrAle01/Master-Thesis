from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import pandas as pd
import torch


@dataclass(frozen=True)
class UtilityEstimationRequest:
    profiles: Sequence[Any]
    full_profile: Any
    docnos: Sequence[str]
    doc_embeddings: torch.Tensor
    loader: Optional[Any] = None
    adapter: Optional[Any] = None
    topics: Optional[pd.DataFrame] = None
    qrels: Optional[pd.DataFrame] = None
    query_embeddings: Optional[torch.Tensor] = None
    corpus_metadata: Optional[pd.DataFrame] = None


@dataclass(frozen=True)
class UtilityEstimationResult:
    utility_table: pd.DataFrame
    pair_details: pd.DataFrame
    report: dict[str, Any]


class UtilityEstimator(ABC):
    requires_query_data: bool = False

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.last_report: dict[str, Any] = {}

    @abstractmethod
    def estimate(self, request: UtilityEstimationRequest) -> UtilityEstimationResult:
        raise NotImplementedError
