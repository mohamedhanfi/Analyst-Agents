"""FileValidator — pure validation logic, no CrewAI imports.

Separated from the CrewAI tool wrapper so the core is unit-testable.
Check order (cheapest first, never reads a big file before cheap gates):
extension -> size (stat-based) -> signature (7 bytes) -> rows + parser (one read).
"""
from __future__ import annotations

from pathlib import Path

import openpyxl
import pandas as pd

from shared.schemas import ValidationError, ValidationResult

SUPPORTED_EXTENSIONS = {".csv", ".xlsx"}

# XLSX = ZIP container; also covers .xls syntax check on second probe
XLSX_MAGIC = b"PK\x03\x04"


class FileValidator:
    def __init__(self, max_file_size_mb: float = 200.0, min_rows: int = 5):
        self.max_file_size_mb = max_file_size_mb
        self.min_rows = min_rows

    # ------------------------------------------------------------------ API

    def validate(self, file_path: str) -> ValidationResult:
        path = Path(file_path)
        size_mb = self._size_mb(path)
        result = ValidationResult(
            file_name=path.name,
            file_path=str(path),
            file_size_mb=round(size_mb, 2),
            validation_status="failed",
        )

        if not path.is_file():
            return self._fail(result, "file_not_found",
                              "File does not exist or is not a regular file.",
                              path.name)

        # 1. Extension -------------------------------------------------------
        extension = path.suffix.lower()
        if extension not in SUPPORTED_EXTENSIONS:
            return self._fail(result, "unsupported_format",
                              "Only CSV and XLSX files are supported.",
                              f"Received extension: {extension or 'none'}")
        result.extension_ok = True

        # 2. File size (independent — never trust caller-passed size) --------
        if size_mb > self.max_file_size_mb:
            return self._fail(result, "file_too_large",
                              f"File exceeds the maximum allowed size of "
                              f"{self.max_file_size_mb} MB.",
                              f"Received size: {size_mb:.2f} MB.")
        result.size_ok = True

        # 3. Signature -------------------------------------------------------
        signature_ok, signature_msg = self._validate_signature(path, extension)
        if not signature_ok:
            return self._fail(result, "invalid_signature",
                              "File content does not match the expected format.",
                              signature_msg)
        result.signature_ok = True

        # 4. Rows + parser (single read for both) ----------------------------
        row_count, parser_ok, parser_msg = self._read_and_parse(path, extension)
        if not parser_ok:
            return self._fail(result, "parser_error",
                              "The file could not be parsed successfully.",
                              parser_msg)
        result.parser_ok = True
        result.row_count = row_count

        # 5. Min rows (§2.1: stop on empty or < 5 rows) -----------------------
        if row_count < self.min_rows:
            return self._fail(result, "min_rows",
                              f"File must contain at least {self.min_rows} rows.",
                              f"Detected {row_count} rows.")
        result.rows_ok = True

        result.validation_status = "passed"
        return result

    # ------------------------------------------------------------ internals

    def _fail(self, result: ValidationResult, code: str, message: str,
              details: str | None) -> ValidationResult:
        result.errors.append(
            ValidationError(code=code, message=message, details=details))
        return result

    @staticmethod
    def _size_mb(path: Path) -> float:
        try:
            return path.stat().st_size / (1024 * 1024)
        except OSError:
            return 0.0

    def _validate_signature(self, path: Path, extension: str):
        try:
            with open(path, "rb") as f:
                first_bytes = f.read(7)
        except OSError as exc:
            return False, f"Unable to read file signature: {exc}"

        if not first_bytes:
            return False, "Empty file."

        if extension == ".csv":
            if b"\x00" in first_bytes:
                return False, "NUL byte detected in the first 7 bytes."
            if first_bytes.startswith(b"\xef\xbb\xbf"):  # UTF-8 BOM
                return True, None
            try:
                first_bytes.decode("utf-8")
                return True, None
            except UnicodeDecodeError:
                return False, "File does not contain valid UTF-8/ASCII text."

        if extension == ".xlsx":
            if first_bytes[:4] != XLSX_MAGIC:
                return False, "Invalid XLSX ZIP signature. Expected PK\\x03\\x04."
            return True, None

        return False, "Unsupported extension."

    def _read_and_parse(self, path: Path, extension: str):
        """Return (row_count, parser_ok, error_msg).

        CSV: a single DataFrame read gives both row count and parse success.
        XLSX: read-only workbook; row count = largest single sheet (the
        pipeline later picks ONE sheet, so summing sheets would mask
        sheets that are too small, violating §2.1).
        """
        if extension == ".csv":
            try:
                df = pd.read_csv(path, encoding="utf-8-sig",
                                 on_bad_lines="error")
                return len(df), True, None
            except UnicodeDecodeError:
                try:
                    df = pd.read_csv(path, encoding="latin-1",
                                     on_bad_lines="error")
                    return len(df), True, None
                except Exception as exc:
                    return 0, False, str(exc)
            except Exception as exc:
                return 0, False, str(exc)

        if extension == ".xlsx":
            try:
                wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
                try:
                    # Same counting as FileReader — openpyxl max_row can be
                    # stale in read_only mode, so count explicitly.
                    rows = max(sum(1 for _ in ws.iter_rows(values_only=True))
                               for ws in wb.worksheets)
                    return rows, True, None
                finally:
                    wb.close()
            except Exception as exc:
                return 0, False, str(exc)

        return 0, False, "Unsupported extension."