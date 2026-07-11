from __future__ import annotations

from pathlib import Path

import pandas as pd


class PairRowsParquetWriter:
    def __init__(self, path: Path, columns: list[str], *, chunk_rows: int = 100_000):
        self.path = Path(path)
        self.columns = columns
        self.chunk_rows = int(chunk_rows)
        self.buffer = []
        self.writer = None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except Exception as exc:
            raise RuntimeError("Streaming score-pair persistence requires `pyarrow`.") from exc
        self.pa = pa
        self.pq = pq

    def add_row(self, row: dict) -> None:
        self.buffer.append(row)
        if len(self.buffer) >= self.chunk_rows:
            self._flush()

    def _flush(self) -> None:
        if not self.buffer:
            return
        frame = pd.DataFrame(self.buffer, columns=self.columns)
        table = self.pa.Table.from_pandas(frame, preserve_index=False)
        if self.writer is None:
            self.writer = self.pq.ParquetWriter(str(self.path), table.schema)
        self.writer.write_table(table)
        self.buffer.clear()

    def close(self) -> None:
        self._flush()
        if self.writer is not None:
            self.writer.close()
        elif not self.path.exists():
            pd.DataFrame(columns=self.columns).to_parquet(self.path, index=False)
