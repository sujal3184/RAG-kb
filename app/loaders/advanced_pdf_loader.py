"""Advanced PDF loader using pdfplumber."""

import io

import pdfplumber

from app.loaders.base import DocumentLoader, LoadedDocument
from app.loaders.exceptions import LoaderError

_COLUMN_GAP_RATIO_THRESHOLD = 0.15


class AdvancedPdfLoader(DocumentLoader):
    """Loads .pdf files with layout-aware, multi-column, and table support."""

    def load(self, content: bytes) -> LoadedDocument:
        """Extract text and tables from a PDF, handling multi-column layouts.

        Raises:
            LoaderError: if the PDF is corrupted, encrypted without a
                usable password, or otherwise unreadable.
        """
        try:
            with pdfplumber.open(io.BytesIO(content)) as pdf:
                page_sections: list[str] = []
                table_count = 0

                for page in pdf.pages:
                    page_text = self._extract_page_text(page)

                    page_tables = page.extract_tables()
                    table_count += len(page_tables)
                    table_texts = [self._format_table(table) for table in page_tables]

                    section = "\n\n".join(filter(None, [page_text, *table_texts]))
                    page_sections.append(section)

                page_count = len(pdf.pages)
        except Exception as exc:
            raise LoaderError(f"Could not open or parse PDF: {exc}") from exc

        full_text = "\n\n".join(page_sections)

        return LoadedDocument(
            text=full_text,
            metadata={"page_count": page_count, "table_count": table_count},
        )

    def _extract_page_text(self, page: "pdfplumber.page.Page") -> str:
        words = page.extract_words()
        if not words:
            return ""

        if self._looks_like_two_columns(words, page.width):
            midpoint = page.width / 2
            left_words = [w for w in words if w["x0"] < midpoint]
            right_words = [w for w in words if w["x0"] >= midpoint]
            return self._words_to_text(left_words) + "\n" + self._words_to_text(right_words)

        return page.extract_text() or ""

    @staticmethod
    def _looks_like_two_columns(words: list[dict], page_width: float) -> bool:
        midpoint = page_width * 0.5
        gap_zone_start = page_width * (0.5 - _COLUMN_GAP_RATIO_THRESHOLD / 2)
        gap_zone_end = page_width * (0.5 + _COLUMN_GAP_RATIO_THRESHOLD / 2)

        words_in_gap_zone = sum(1 for w in words if gap_zone_start < w["x0"] < gap_zone_end)
        words_left_of_mid = sum(1 for w in words if w["x0"] < midpoint)
        words_right_of_mid = sum(1 for w in words if w["x0"] >= midpoint)

        has_substantial_content_both_sides = words_left_of_mid > 5 and words_right_of_mid > 5
        gap_is_mostly_empty = words_in_gap_zone < (len(words) * 0.05)

        return has_substantial_content_both_sides and gap_is_mostly_empty

    @staticmethod
    def _words_to_text(words: list[dict]) -> str:
        if not words:
            return ""

        lines: list[list[dict]] = []
        current_line: list[dict] = [words[0]]

        for word in words[1:]:
            same_line_as_previous = abs(word["top"] - current_line[-1]["top"]) < 3
            if same_line_as_previous:
                current_line.append(word)
            else:
                lines.append(current_line)
                current_line = [word]
        lines.append(current_line)

        return "\n".join(" ".join(w["text"] for w in line) for line in lines)

    @staticmethod
    def _format_table(table: list[list[str | None]]) -> str:
        if not table or not table[0]:
            return ""

        headers = [cell or "" for cell in table[0]]
        rows_text = []
        for row in table[1:]:
            cells = [cell or "" for cell in row]
            formatted = ", ".join(
                f"{header}: {value}" for header, value in zip(headers, cells, strict=False)
            )
            rows_text.append(formatted)

        return "\n".join(rows_text)