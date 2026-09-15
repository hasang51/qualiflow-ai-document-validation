"""Isolated, optional legacy OCR utility.

This module is NOT part of the main extraction runtime. The primary extraction
path in ``app/services/extraction_pipeline.py`` is Bedrock Gemma multimodal and
does not import anything from this file.

It is kept for offline experimentation via ``scripts/check_ocr_backend.py``
and to preserve the deterministic cell-level parsing helpers used by the
tests. The Tesseract backend imports ``pytesseract`` lazily inside a
try/except so the package and the Tesseract binary are both optional; the app
starts and tests run without either one installed.
"""

from __future__ import annotations

import logging
import shutil
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np

from app.config import settings

OCRStrategy = Literal["alphanumeric", "numeric", "fallback"]

NUMERIC_FIELDS = {"yield_strength_mpa", "tensile_strength_mpa", "elongation_percentage"}
ALPHANUMERIC_FIELDS = {"heat_number", "item_id", "grade", "weight_or_length"}
TARGET_FIELDS = NUMERIC_FIELDS | ALPHANUMERIC_FIELDS
FIELD_FAMILY_VARIANT_QUALITY: dict[str, dict[str, float]] = {
    "alphanumeric": {
        "alpha_scale2_stroke_preserve": 0.92,
        "alpha_scale3_line_suppressed": 0.88,
        "alpha_scale2_moderate_threshold": 0.80,
    },
    "numeric": {
        "numeric_scale2_local_threshold": 0.90,
        "numeric_scale3_local_threshold": 0.95,
        "numeric_scale3_line_suppressed": 0.86,
    },
    "generic": {"generic_scale2_otsu": 0.75},
}

logger = logging.getLogger("qualiflow.ocr")


@dataclass(frozen=True)
class NormalizationDecision:
    raw_text: str
    normalized_text: str
    normalization_applied: bool
    normalization_notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_text": self.raw_text,
            "normalized_text": self.normalized_text,
            "normalization_applied": self.normalization_applied,
            "normalization_notes": self.normalization_notes,
        }


@dataclass(frozen=True)
class ParsedCellValue:
    canonical_field: str | None
    parsed_value: str | float | None
    unresolved: bool
    unresolved_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonical_field": self.canonical_field,
            "parsed_value": self.parsed_value,
            "unresolved": self.unresolved,
            "unresolved_reason": self.unresolved_reason,
        }


@dataclass(frozen=True)
class OCRCellResult:
    row_index: int
    column_index: int
    canonical_field: str | None
    image_path: str
    strategy: OCRStrategy
    preprocessing_family: str
    variant_name: str
    attempted_passes: list[dict[str, Any]]
    candidate_score: float
    score_breakdown: dict[str, Any]
    geometry_consistent: bool
    raw_text: str
    normalized_text: str
    parsed_value: str | float | None
    ocr_confidence: float | None
    normalization_applied: bool
    normalization_notes: list[str]
    final_value: str | float | None
    unresolved: bool
    unresolved_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "row_index": self.row_index,
            "column_index": self.column_index,
            "canonical_field": self.canonical_field,
            "image_path": self.image_path,
            "strategy": self.strategy,
            "preprocessing_family": self.preprocessing_family,
            "variant_name": self.variant_name,
            "attempted_passes": self.attempted_passes,
            "candidate_score": self.candidate_score,
            "score_breakdown": self.score_breakdown,
            "geometry_consistent": self.geometry_consistent,
            "raw_text": self.raw_text,
            "normalized_text": self.normalized_text,
            "parsed_value": self.parsed_value,
            "ocr_confidence": self.ocr_confidence,
            "normalization_applied": self.normalization_applied,
            "normalization_notes": self.normalization_notes,
            "final_value": self.final_value,
            "unresolved": self.unresolved,
            "unresolved_reason": self.unresolved_reason,
        }


@dataclass(frozen=True)
class OCRRunResult:
    backend: str
    backend_available: bool
    backend_diagnostics: dict[str, Any]
    page_results: list[dict[str, Any]]
    cell_results: list[OCRCellResult]
    unresolved_count: int
    resolved_count: int
    summary_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "backend_available": self.backend_available,
            "backend_diagnostics": self.backend_diagnostics,
            "page_results": self.page_results,
            "cell_results": [cell.to_dict() for cell in self.cell_results],
            "unresolved_count": self.unresolved_count,
            "resolved_count": self.resolved_count,
            "summary_notes": self.summary_notes,
        }


class BaseOCRBackend:
    name = "base"

    def is_available(self) -> bool:
        return False

    def availability_diagnostics(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "available": False,
            "reason": "OCR backend not implemented",
            "expected_fix": "Configure OCR_BACKEND to a supported backend.",
        }

    def read_text(self, image: np.ndarray, strategy: OCRStrategy) -> tuple[str, float | None]:
        raise NotImplementedError


class TesseractOCRBackend(BaseOCRBackend):
    name = "pytesseract"

    def __init__(self) -> None:
        self._import_error: str | None = None
        self._diagnostics_cache: dict[str, Any] | None = None
        self._configured_cmd = settings.tesseract_cmd.strip()
        try:
            import pytesseract  # type: ignore
        except Exception as exc:
            self._pytesseract = None
            self._import_error = str(exc)
        else:
            self._pytesseract = pytesseract

    def _candidate_binary_paths(self) -> list[str]:
        candidates = [
            self._configured_cmd,
            shutil.which("tesseract") or "",
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        ]
        unique: list[str] = []
        for path in candidates:
            path = (path or "").strip()
            if path and path not in unique:
                unique.append(path)
        return unique

    def _resolve_executable(self) -> tuple[str | None, list[str]]:
        tried = self._candidate_binary_paths()
        for candidate in tried:
            if Path(candidate).exists():
                return candidate, tried
        return None, tried

    def availability_diagnostics(self) -> dict[str, Any]:
        if self._diagnostics_cache is not None:
            return self._diagnostics_cache

        package_installed = self._pytesseract is not None
        if not package_installed:
            self._diagnostics_cache = {
                "backend": self.name,
                "available": False,
                "reason": "Python package pytesseract is not importable",
                "import_error": self._import_error,
                "tesseract_cmd_configured": self._configured_cmd or None,
                "expected_fix": "Run 'pip install pytesseract' in the active virtual environment.",
            }
            return self._diagnostics_cache

        resolved_cmd, candidates = self._resolve_executable()
        if resolved_cmd:
            try:
                self._pytesseract.pytesseract.tesseract_cmd = resolved_cmd
            except Exception:
                pass

        try:
            version = str(self._pytesseract.get_tesseract_version())
            available = True
            reason = "pytesseract package and tesseract executable are available"
            expected_fix = None
        except Exception as exc:
            version = None
            available = False
            if not resolved_cmd:
                reason = "Tesseract executable not found"
                expected_fix = (
                    "Install Tesseract OCR and set TESSERACT_CMD to the full executable path, "
                    "for example C:\\Program Files\\Tesseract-OCR\\tesseract.exe."
                )
            else:
                reason = f"Tesseract binary is configured but not runnable: {exc}"
                expected_fix = "Verify TESSERACT_CMD points to a valid executable and restart the API."

        self._diagnostics_cache = {
            "backend": self.name,
            "available": available,
            "reason": reason,
            "expected_fix": expected_fix,
            "package_installed": package_installed,
            "tesseract_version": version,
            "tesseract_cmd_configured": self._configured_cmd or None,
            "tesseract_cmd_effective": resolved_cmd,
            "binary_candidates_checked": candidates,
        }
        return self._diagnostics_cache

    def is_available(self) -> bool:
        return bool(self.availability_diagnostics().get("available"))

    def read_text(self, image: np.ndarray, strategy: OCRStrategy) -> tuple[str, float | None]:
        if self._pytesseract is None:
            return "", None
        try:
            cfg = "--oem 3 --psm 7"
            if strategy == "numeric":
                cfg += " -c tessedit_char_whitelist=0123456789.,"
            elif strategy == "alphanumeric":
                cfg += " -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-./%"
            data = self._pytesseract.image_to_data(
                image,
                config=cfg,
                output_type=self._pytesseract.Output.DICT,
                timeout=2,
            )
            tokens = []
            confidences: list[float] = []
            for text, conf in zip(data.get("text", []), data.get("conf", [])):
                if not str(text).strip():
                    continue
                tokens.append(str(text).strip())
                try:
                    conf_val = float(conf)
                except (TypeError, ValueError):
                    continue
                if conf_val >= 0:
                    confidences.append(conf_val / 100.0)
            joined = " ".join(tokens).strip()
            avg_conf = (sum(confidences) / len(confidences)) if confidences else None
            return joined, avg_conf
        except Exception:
            return "", None


def select_ocr_strategy(canonical_field: str | None) -> OCRStrategy:
    if canonical_field in NUMERIC_FIELDS:
        return "numeric"
    if canonical_field in ALPHANUMERIC_FIELDS:
        return "alphanumeric"
    return "fallback"


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_ocr_text(
    *,
    canonical_field: str | None,
    raw_text: str,
    confidence: float | None,
) -> NormalizationDecision:
    notes: list[str] = []
    normalized = _normalize_whitespace(raw_text)
    original = normalized

    if canonical_field in NUMERIC_FIELDS:
        normalized = normalized.replace(" ", "")
        if "," in normalized and "." not in normalized:
            normalized = normalized.replace(",", ".")
            notes.append("converted decimal comma to dot")
        if re.search(r"[OolISs]", normalized):
            if confidence is not None and confidence >= 0.70:
                candidate = normalized.translate(str.maketrans({"O": "0", "o": "0", "I": "1", "l": "1", "S": "5", "s": "5"}))
                if re.fullmatch(r"[0-9.]+", candidate):
                    normalized = candidate
                    notes.append("applied conservative OCR confusion repair for numeric field")
                else:
                    notes.append("confusion repair skipped due to non-numeric output")
            else:
                notes.append("confusion-like characters kept due to low confidence")
    elif canonical_field in ALPHANUMERIC_FIELDS:
        normalized = normalized.upper()
        normalized = re.sub(r"[^A-Z0-9\-./% ]+", "", normalized)
        normalized = _normalize_whitespace(normalized)
    else:
        normalized = _normalize_whitespace(normalized)

    return NormalizationDecision(
        raw_text=raw_text,
        normalized_text=normalized,
        normalization_applied=(normalized != original or bool(notes)),
        normalization_notes=notes,
    )


def parse_heat_number(raw: str, normalized: str) -> str | None:
    candidate = normalized.replace(" ", "")
    if len(candidate) < 4 or not re.fullmatch(r"[A-Z0-9\-./]+", candidate):
        return None
    return candidate


def parse_item_id(raw: str, normalized: str) -> str | None:
    candidate = normalized.replace(" ", "")
    if len(candidate) < 3 or not re.fullmatch(r"[A-Z0-9\-./]+", candidate):
        return None
    return candidate


def parse_grade(raw: str, normalized: str) -> str | None:
    candidate = normalized.replace(" ", "")
    if len(candidate) < 3:
        return None
    if not re.fullmatch(r"[A-Z0-9+.\-/]+", candidate):
        return None
    return candidate


def parse_numeric_strength(raw: str, normalized: str) -> float | None:
    candidate = normalized.strip()
    if not re.fullmatch(r"\d+(?:\.\d+)?", candidate):
        return None
    value = float(candidate)
    if value <= 0 or value > 2000:
        return None
    return value


def parse_percentage(raw: str, normalized: str) -> float | None:
    candidate = normalized.replace("%", "").strip()
    if not re.fullmatch(r"\d+(?:\.\d+)?", candidate):
        return None
    value = float(candidate)
    if value < 0 or value > 100:
        return None
    return value


def _parse_by_field(canonical_field: str | None, raw: str, normalized: str) -> ParsedCellValue:
    if canonical_field == "heat_number":
        parsed = parse_heat_number(raw, normalized)
    elif canonical_field == "item_id":
        parsed = parse_item_id(raw, normalized)
    elif canonical_field == "grade":
        parsed = parse_grade(raw, normalized)
    elif canonical_field in {"yield_strength_mpa", "tensile_strength_mpa"}:
        parsed = parse_numeric_strength(raw, normalized)
    elif canonical_field == "elongation_percentage":
        parsed = parse_percentage(raw, normalized)
    elif canonical_field == "weight_or_length":
        parsed = normalized if normalized else None
    else:
        parsed = None

    if parsed is None:
        return ParsedCellValue(
            canonical_field=canonical_field,
            parsed_value=None,
            unresolved=True,
            unresolved_reason="field parser could not produce a safe value",
        )
    return ParsedCellValue(
        canonical_field=canonical_field,
        parsed_value=parsed,
        unresolved=False,
        unresolved_reason=None,
    )


def _prepare_cell_image(image_path: str) -> np.ndarray | None:
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    return image


def _tight_crop_with_padding(image: np.ndarray, pad: int = 2) -> np.ndarray:
    inv = 255 - image
    ys, xs = np.where(inv > 20)
    if len(xs) == 0 or len(ys) == 0:
        return image
    x1 = max(0, int(xs.min()) - pad)
    y1 = max(0, int(ys.min()) - pad)
    x2 = min(image.shape[1], int(xs.max()) + pad + 1)
    y2 = min(image.shape[0], int(ys.max()) + pad + 1)
    cropped = image[y1:y2, x1:x2]
    return cropped if cropped.size > 0 else image


def _suppress_table_lines(binary: np.ndarray) -> np.ndarray:
    inv = 255 - binary
    h, w = binary.shape[:2]
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(8, h // 2)))
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(8, w // 2), 1))
    vertical = cv2.morphologyEx(inv, cv2.MORPH_OPEN, vertical_kernel)
    horizontal = cv2.morphologyEx(inv, cv2.MORPH_OPEN, horizontal_kernel)
    lines = cv2.bitwise_or(vertical, horizontal)
    cleaned_inv = cv2.bitwise_and(inv, cv2.bitwise_not(lines))
    return 255 - cleaned_inv


def _field_preprocessing_family(canonical_field: str | None) -> str:
    if canonical_field in {"heat_number", "item_id", "grade"}:
        return "alphanumeric"
    if canonical_field in NUMERIC_FIELDS:
        return "numeric"
    return "generic"


def _numeric_cleanup(binary: np.ndarray) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    return opened


def _build_cell_variants(image: np.ndarray, canonical_field: str | None) -> tuple[str, dict[str, np.ndarray]]:
    family = _field_preprocessing_family(canonical_field)
    base = _tight_crop_with_padding(image, pad=1 if family == "numeric" else 2)
    variants: dict[str, np.ndarray] = {}
    if family == "alphanumeric":
        scale2 = cv2.resize(base, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        blur2 = cv2.GaussianBlur(scale2, (3, 3), 0)
        preserve = cv2.addWeighted(scale2, 1.35, blur2, -0.35, 0)
        _, preserve_otsu = cv2.threshold(preserve, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants["alpha_scale2_stroke_preserve"] = preserve_otsu

        scale3 = cv2.resize(base, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
        sharpen = cv2.filter2D(scale3, -1, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]))
        _, sharpen_otsu = cv2.threshold(sharpen, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants["alpha_scale3_line_suppressed"] = _suppress_table_lines(sharpen_otsu)

        moderate = cv2.adaptiveThreshold(
            scale2,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            11,
        )
        variants["alpha_scale2_moderate_threshold"] = moderate
    elif family == "numeric":
        scale2 = cv2.resize(base, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        scale3 = cv2.resize(base, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
        for name, arr in (
            ("numeric_scale2_local_threshold", scale2),
            ("numeric_scale3_local_threshold", scale3),
        ):
            blur = cv2.GaussianBlur(arr, (3, 3), 0)
            adaptive = cv2.adaptiveThreshold(
                blur,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                29,
                7,
            )
            variants[name] = _numeric_cleanup(adaptive)
        variants["numeric_scale3_line_suppressed"] = _suppress_table_lines(variants["numeric_scale3_local_threshold"])
    else:
        scale2 = cv2.resize(base, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        _, otsu = cv2.threshold(scale2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants["generic_scale2_otsu"] = otsu
    return family, variants


def _select_strategies_for_field(canonical_field: str | None) -> list[OCRStrategy]:
    primary = select_ocr_strategy(canonical_field)
    if primary == "numeric":
        return ["numeric", "fallback"]
    if primary == "alphanumeric":
        return ["alphanumeric", "fallback"]
    return ["fallback", "alphanumeric", "numeric"]


def _field_compatibility_score(canonical_field: str | None, normalized_text: str, parsed_value: Any) -> float:
    if parsed_value is not None:
        return 1.0
    text = (normalized_text or "").strip()
    if not text:
        return 0.0
    if canonical_field in NUMERIC_FIELDS:
        allowed = sum(1 for char in text if char.isdigit() or char in ".,%")
        return allowed / max(len(text), 1)
    if canonical_field in ALPHANUMERIC_FIELDS:
        allowed = sum(1 for char in text if char.isalnum() or char in "-./% ")
        return allowed / max(len(text), 1)
    return 0.5


def _text_quality_score(normalized_text: str) -> float:
    text = (normalized_text or "").strip()
    if not text:
        return 0.0
    alnum = sum(1 for char in text if char.isalnum())
    dense = alnum / max(len(text), 1)
    length_factor = min(1.0, len(text) / 8.0)
    return round((dense * 0.7) + (length_factor * 0.3), 4)


def _variant_quality_score(preprocessing_family: str, variant_name: str) -> float:
    return FIELD_FAMILY_VARIANT_QUALITY.get(preprocessing_family, {}).get(variant_name, 0.5)


def _score_ocr_candidate(
    *,
    canonical_field: str | None,
    preprocessing_family: str,
    variant_name: str,
    unresolved: bool,
    parsed_value: str | float | None,
    confidence: float | None,
    normalized_text: str,
    geometry_consistent: bool,
) -> dict[str, Any]:
    parse_success = 1.0 if (not unresolved and parsed_value is not None) else 0.0
    field_compatibility = _field_compatibility_score(canonical_field, normalized_text, parsed_value)
    confidence_score = max(0.0, min(1.0, float(confidence))) if confidence is not None else 0.0
    text_quality = _text_quality_score(normalized_text)
    geometry_score = 1.0 if geometry_consistent else 0.0
    variant_quality = _variant_quality_score(preprocessing_family, variant_name)
    total = round(
        (parse_success * 0.35)
        + (field_compatibility * 0.20)
        + (confidence_score * 0.15)
        + (text_quality * 0.10)
        + (geometry_score * 0.10)
        + (variant_quality * 0.10),
        4,
    )
    return {
        "total": total,
        "parse_success": parse_success,
        "field_compatibility": round(field_compatibility, 4),
        "confidence": round(confidence_score, 4),
        "text_quality": round(text_quality, 4),
        "geometry_consistency": geometry_score,
        "variant_quality": round(variant_quality, 4),
    }


def create_ocr_backend() -> BaseOCRBackend:
    backend_name = settings.ocr_backend
    if backend_name in {"pytesseract", "tesseract"}:
        return TesseractOCRBackend()
    return BaseOCRBackend()


def get_ocr_backend_diagnostics(backend: BaseOCRBackend | None = None) -> dict[str, Any]:
    backend = backend or create_ocr_backend()
    diagnostics = backend.availability_diagnostics()
    diagnostics["configured_backend"] = settings.ocr_backend
    return diagnostics


def log_ocr_backend_diagnostics() -> dict[str, Any]:
    diagnostics = get_ocr_backend_diagnostics()
    if diagnostics.get("available"):
        logger.info(
            "OCR backend ready: backend=%s version=%s cmd=%s",
            diagnostics.get("backend"),
            diagnostics.get("tesseract_version"),
            diagnostics.get("tesseract_cmd_effective"),
        )
    else:
        logger.warning(
            "OCR backend unavailable: backend=%s reason=%s expected_fix=%s configured_cmd=%s",
            diagnostics.get("backend"),
            diagnostics.get("reason"),
            diagnostics.get("expected_fix"),
            diagnostics.get("tesseract_cmd_configured"),
        )
    return diagnostics


def run_ocr_from_table_parsing(preprocessing_meta: dict[str, Any], backend: BaseOCRBackend | None = None) -> OCRRunResult:
    backend = backend or create_ocr_backend()
    backend_diag = backend.availability_diagnostics()
    backend_available = bool(backend_diag.get("available"))
    if settings.ocr_fail_loudly and not backend_available:
        reason = backend_diag.get("reason") or "OCR backend unavailable"
        expected_fix = backend_diag.get("expected_fix")
        if expected_fix:
            reason = f"{reason}. {expected_fix}"
        raise RuntimeError(reason)
    page_results: list[dict[str, Any]] = []
    cells: list[OCRCellResult] = []

    for page in preprocessing_meta.get("pages", []):
        parse = (page or {}).get("table_parsing") or {}
        parse_cells = parse.get("cell_crops") or []
        page_out = {
            "page": page.get("page"),
            "cell_count": len(parse_cells),
            "ocr_processed_cells": 0,
            "ocr_resolved_cells": 0,
            "ocr_unresolved_cells": 0,
        }
        for cell in parse_cells:
            canonical_field = cell.get("canonical_field")
            if canonical_field not in TARGET_FIELDS:
                continue
            strategies = _select_strategies_for_field(canonical_field)
            image_path = str(cell.get("image_path") or "")
            source_image = _prepare_cell_image(image_path)
            geometry_consistent = bool(
                canonical_field in TARGET_FIELDS
                and int(cell.get("row_index") or 0) > 0
                and int(cell.get("column_index") or 0) >= 0
            )
            if source_image is None:
                result = OCRCellResult(
                    row_index=int(cell.get("row_index") or 0),
                    column_index=int(cell.get("column_index") or 0),
                    canonical_field=canonical_field,
                    image_path=image_path,
                    strategy=strategies[0],
                    preprocessing_family=_field_preprocessing_family(canonical_field),
                    variant_name="missing",
                    attempted_passes=[],
                    candidate_score=0.0,
                    score_breakdown={"total": 0.0},
                    geometry_consistent=geometry_consistent,
                    raw_text="",
                    normalized_text="",
                    parsed_value=None,
                    ocr_confidence=None,
                    normalization_applied=False,
                    normalization_notes=[],
                    final_value=None,
                    unresolved=True,
                    unresolved_reason="cell image is missing or unreadable",
                )
            elif not backend_available:
                result = OCRCellResult(
                    row_index=int(cell.get("row_index") or 0),
                    column_index=int(cell.get("column_index") or 0),
                    canonical_field=canonical_field,
                    image_path=image_path,
                    strategy=strategies[0],
                    preprocessing_family=_field_preprocessing_family(canonical_field),
                    variant_name="unavailable",
                    attempted_passes=[],
                    candidate_score=0.0,
                    score_breakdown={"total": 0.0},
                    geometry_consistent=geometry_consistent,
                    raw_text="",
                    normalized_text="",
                    parsed_value=None,
                    ocr_confidence=None,
                    normalization_applied=False,
                    normalization_notes=[],
                    final_value=None,
                    unresolved=True,
                    unresolved_reason=str(backend_diag.get("reason") or "OCR backend unavailable"),
                )
            else:
                preprocessing_family, variants = _build_cell_variants(source_image, canonical_field)
                attempts: list[dict[str, Any]] = []
                best_payload: dict[str, Any] | None = None
                best_score = -1.0
                stop_early = False

                primary_strategy = strategies[0]
                secondary_strategy = strategies[1] if len(strategies) > 1 else strategies[0]
                plan: list[tuple[str, OCRStrategy]] = []
                for variant_name in variants.keys():
                    plan.append((variant_name, primary_strategy))
                if secondary_strategy != primary_strategy and plan:
                    plan.append((plan[0][0], secondary_strategy))

                for variant_name, strategy in plan:
                    variant_img = variants[variant_name]
                    raw_text, conf = backend.read_text(variant_img, strategy=strategy)
                    if raw_text.strip():
                        decision = normalize_ocr_text(
                            canonical_field=canonical_field,
                            raw_text=raw_text,
                            confidence=conf,
                        )
                        parsed = _parse_by_field(canonical_field, raw_text, decision.normalized_text)
                    else:
                        decision = NormalizationDecision(
                            raw_text="",
                            normalized_text="",
                            normalization_applied=False,
                            normalization_notes=[],
                        )
                        parsed = ParsedCellValue(
                            canonical_field=canonical_field,
                            parsed_value=None,
                            unresolved=True,
                            unresolved_reason="empty OCR output",
                        )
                    score_breakdown = _score_ocr_candidate(
                        canonical_field=canonical_field,
                        preprocessing_family=preprocessing_family,
                        variant_name=variant_name,
                        unresolved=parsed.unresolved,
                        parsed_value=parsed.parsed_value,
                        confidence=conf,
                        normalized_text=decision.normalized_text,
                        geometry_consistent=geometry_consistent,
                    )
                    attempts.append(
                        {
                            "variant_name": variant_name,
                            "strategy": strategy,
                            "raw_text": raw_text,
                            "normalized_text": decision.normalized_text,
                            "parsed_value": parsed.parsed_value,
                            "ocr_confidence": conf,
                            "unresolved": parsed.unresolved,
                            "unresolved_reason": parsed.unresolved_reason,
                            "score_breakdown": score_breakdown,
                            "candidate_score": score_breakdown["total"],
                        }
                    )
                    if score_breakdown["total"] > best_score:
                        best_score = score_breakdown["total"]
                        best_payload = {
                            "variant_name": variant_name,
                            "strategy": strategy,
                            "raw_text": raw_text,
                            "decision": decision,
                            "parsed": parsed,
                            "conf": conf,
                            "score_breakdown": score_breakdown,
                        }
                    if (
                        not parsed.unresolved
                        and parsed.parsed_value is not None
                        and score_breakdown["total"] >= 0.82
                    ):
                        stop_early = True
                        break
                if stop_early:
                    pass

                if best_payload is None:
                    best_payload = {
                        "variant_name": "none",
                        "strategy": strategies[0],
                        "raw_text": "",
                        "decision": NormalizationDecision("", "", False, []),
                        "parsed": ParsedCellValue(canonical_field, None, True, "empty OCR output"),
                        "conf": None,
                        "score_breakdown": {"total": 0.0},
                    }

                decision = best_payload["decision"]
                parsed = best_payload["parsed"]
                result = OCRCellResult(
                    row_index=int(cell.get("row_index") or 0),
                    column_index=int(cell.get("column_index") or 0),
                    canonical_field=canonical_field,
                    image_path=image_path,
                    strategy=best_payload["strategy"],
                    preprocessing_family=preprocessing_family,
                    variant_name=best_payload["variant_name"],
                    attempted_passes=attempts,
                    candidate_score=float(best_payload["score_breakdown"]["total"]),
                    score_breakdown=best_payload["score_breakdown"],
                    geometry_consistent=geometry_consistent,
                    raw_text=best_payload["raw_text"],
                    normalized_text=decision.normalized_text,
                    parsed_value=parsed.parsed_value,
                    ocr_confidence=best_payload["conf"],
                    normalization_applied=decision.normalization_applied,
                    normalization_notes=decision.normalization_notes,
                    final_value=parsed.parsed_value,
                    unresolved=parsed.unresolved,
                    unresolved_reason=parsed.unresolved_reason,
                )
            page_out["ocr_processed_cells"] += 1
            if result.unresolved:
                page_out["ocr_unresolved_cells"] += 1
            else:
                page_out["ocr_resolved_cells"] += 1
            cells.append(result)
        page_results.append(page_out)

    resolved = sum(1 for cell in cells if not cell.unresolved)
    unresolved = sum(1 for cell in cells if cell.unresolved)
    notes: list[str] = []
    if not backend_available:
        notes.append(f"OCR backend unavailable: {backend_diag.get('reason')}")
        if backend_diag.get("expected_fix"):
            notes.append(str(backend_diag.get("expected_fix")))
    if not cells:
        notes.append("No cell crops available from table parsing")
    return OCRRunResult(
        backend=backend.name,
        backend_available=backend_available,
        backend_diagnostics=backend_diag,
        page_results=page_results,
        cell_results=cells,
        unresolved_count=unresolved,
        resolved_count=resolved,
        summary_notes=notes,
    )

