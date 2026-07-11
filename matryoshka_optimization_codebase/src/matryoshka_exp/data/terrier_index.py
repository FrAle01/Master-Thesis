from __future__ import annotations

from pathlib import Path

import pandas as pd


class TerrierIndexResolver:
    def __init__(self, data_cfg, pt, dataset, logger):
        self.data_cfg = data_cfg
        self.pt = pt
        self.dataset = dataset
        self.logger = logger

    def resolve(self):
        """
        Try built-in dataset index first; if unavailable, fall back to a local index.
        """
        if self.dataset is not None:
            try:
                built_in = self.dataset.get_index()
                if built_in is not None:
                    return built_in
            except (AttributeError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
                self.logger.warning(
                    "Failed to use built-in Terrier index for dataset %s; falling back to local index. Error: %s",
                    self.data_cfg.pyterrier_dataset,
                    exc,
                )

        return self._load_or_build_local_index()

    def _load_or_build_local_index(self):
        index_path = self._default_index_path().resolve()
        data_properties = index_path / "data.properties"
        text_fields = [field for field in self.data_cfg.text_fields if str(field).strip()]

        # Reuse an existing local Terrier index if present.
        if data_properties.exists() and not self.data_cfg.terrier_index_overwrite:
            index_ref = self.pt.IndexRef.of(str(index_path))
            index_obj = self.pt.IndexFactory.of(index_ref)
            num_fields = int(index_obj.getCollectionStatistics().getNumberOfFields())
            if num_fields > 0:
                return index_ref
            self.logger.warning(
                "Existing Terrier index at %s has no fields; rebuilding with fields enabled.",
                index_path,
            )

        if not self.data_cfg.build_local_terrier_index_if_missing:
            raise RuntimeError(
                f"No built-in Terrier index is available and no local index was found at {index_path}."
            )
        if not text_fields:
            raise ValueError(
                "data.text_fields must contain at least one non-empty field to build a Terrier index."
            )

        index_path.mkdir(parents=True, exist_ok=True)

        meta = self.data_cfg.terrier_meta_lengths or {
            "docno": 64,
            **{field: 4096 for field in text_fields},
        }

        indexer = self.pt.IterDictIndexer(
            str(index_path),
            meta=meta,
            text_attrs=text_fields,
            fields=True,
            threads=self.data_cfg.terrier_index_threads,
            overwrite=True,
        )

        source_iter = self._build_iterdict_source()
        index_ref = indexer.index(source_iter)
        return index_ref

    def _build_iterdict_source(self):
        """
        Return an iterator of dicts suitable for pt.IterDictIndexer.
        We make the output explicit so it works for both ir_datasets corpora
        and local corpora.
        """
        if self.data_cfg.local_corpus_path:
            df = pd.read_parquet(self.data_cfg.local_corpus_path)
            emitted = 0
            for row in df.to_dict(orient="records"):
                out = {"docno": str(row[self.data_cfg.docno_column])}
                for field in self.data_cfg.text_fields:
                    out[field] = str(row.get(field, "") or "")
                yield out
                emitted += 1
                if self.data_cfg.max_docs is not None and emitted >= self.data_cfg.max_docs:
                    break
            return

        emitted = 0
        for record in self.dataset.get_corpus_iter(verbose=True):
            out = {"docno": str(record[self.data_cfg.docno_column])}
            for field in self.data_cfg.text_fields:
                out[field] = str(record.get(field, "") or "")
            yield out
            emitted += 1
            if self.data_cfg.max_docs is not None and emitted >= self.data_cfg.max_docs:
                break

    def _default_index_path(self) -> Path:
        if self.data_cfg.local_terrier_index_path:
            return Path(self.data_cfg.local_terrier_index_path)
        if self.data_cfg.pyterrier_dataset:
            safe_name = self.data_cfg.pyterrier_dataset.replace(":", "_").replace("/", "_")
            return Path("./indices") / safe_name
        return Path("./indices/local_corpus")
