"""Loader for CSV files (.csv).

Converts each row into a readable "column: value" sentence rather than
dumping raw comma-separated text — this produces far more meaningful
chunks for embedding and retrieval than raw CSV syntax would.
"""

import csv
import io

from app.loaders.base import DocumentLoader, LoadedDocument
from app.loaders.exceptions import LoaderError


class CsvLoader(DocumentLoader):
    """Loads .csv files, converting each row into a readable text line."""

    def load(self, content: bytes) -> LoadedDocument:
        """Parse CSV rows and format each as 'column: value, column: value'.

        Raises:
            LoaderError: if the content can't be decoded or has no rows at all.
        """
        try:
            decoded = content.decode("utf-8", errors="replace")
            reader = csv.DictReader(io.StringIO(decoded))
            rows = list(reader)
        except csv.Error as exc:
            raise LoaderError(f"Could not parse CSV content: {exc}") from exc

        if not rows:
            raise LoaderError("CSV file has no data rows")

        lines = []
        for row in rows:
            formatted = ", ".join(f"{key}: {value}" for key, value in row.items() if key)
            lines.append(formatted)

        return LoadedDocument(
            text="\n".join(lines),
            metadata={
                "row_count": len(rows),
                "column_names": list(rows[0].keys()) if rows else [],
            },
        )