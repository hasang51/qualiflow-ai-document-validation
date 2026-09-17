"""Generic mechanical-property table row/column alignment for MTC tables.

Mill-test mechanical tables often list one property per row with multiple
value columns (Specified, Min, Max, Results). LLM extraction frequently
aligns the wrong column — for example taking a minimum or specification value
instead of the observed Results column. This module maps labeled table rows
into canonical mechanical properties using property-label and column-header
semantics only (no supplier-specific rules).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

from app.domain.field_mapping_registry import normalize_header
from app.domain.numeric_parser import parse_numeric

RESULT_COLUMN_ALIASES: frozenset[str] = frozenset(
    {
        "RESULTS",
        "RESULT",
        "ACTUAL",
        "OBSERVED",
        "TEST RESULT",
        "TEST RESULTS",
        "MEASURED",
        "VALUE",
    }
)

VALIDATION_COLUMN_ALIASES: frozenset[str] = frozenset(
    {
        "SPECIFIED",
        "SPEC",
        "SPECIFICATION",
        "REQUIREMENT",
        "REQUIREMENTS",
        "MIN",
        "MINIMUM",
        "MAX",
        "MAXIMUM",
        "LIMIT",
        "LIMITS",
        "STANDARD",
        "NORM",
        "NORMATIVE",
        "REQ",
    }
)

_ROW_META_KEYS: frozenset[str] = frozenset(
    {
        "PROPERTY",
        "PROPERTY LABEL",
        "PROPERTY_LABEL",
        "CHARACTERISTIC",
        "TEST",
        "PARAMETER",
        "ITEM ID",
        "ITEM_ID",
        "GRADE",
        "HEAT NUMBER",
        "HEAT_NUMBER",
        "ROW CONFIDENCE",
        "ROW_CONFIDENCE",
        "NEEDS REVIEW",
        "NEEDS_REVIEW",
        "FIELD CONFIDENCE",
        "FIELD_CONFIDENCE",
        "MECHANICAL PROPERTIES",
        "MECHANICAL_PROPERTIES",
        "MECHANICAL TABLE ROWS",
        "MECHANICAL_TABLE_ROWS",
        "CHEMICAL COMPOSITION",
        "CHEMICAL_COMPOSITION",
        "CHEMICAL TABLE ROWS",
        "CHEMICAL_TABLE_ROWS",
        "SOURCE PAGE",
        "SOURCE_PAGE",
        "DIMENSIONS",
        "STANDARDS",
    }
)

_TENSILE_EXCLUDE_MARKERS: tuple[str, ...] = (
    "PROOF",
    "YIELD",
    "RP0",
    "RP1",
    "HARDNESS",
    "IMPACT",
    "CHARPY",
    "IZOD",
    "HB",
    "HV",
    "HRC",
    "BRINELL",
    "VICKERS",
)

_PROPERTY_LABEL_KEYS: tuple[str, ...] = (
    "property",
    "property_label",
    "characteristic",
    "test",
    "parameter",
    "item_id",
    "grade",
)


@dataclass
class MechanicalTableMappingResult:
    mechanical_properties: dict[str, float | None] = field(default_factory=dict)
    auxiliary: dict[str, Any] = field(default_factory=dict)
    uncertain: bool = False
    tokens: list[str] = field(default_factory=list)
    trace: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mechanical_properties": dict(self.mechanical_properties),
            "auxiliary": dict(self.auxiliary),
            "uncertain": self.uncertain,
            "tokens": list(self.tokens),
            "trace": dict(self.trace),
        }


def _text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _normalize_column_key(key: str) -> str:
    return normalize_header(str(key))


def _column_kind(normalized_key: str) -> str | None:
    if not normalized_key:
        return None
    if normalized_key in RESULT_COLUMN_ALIASES:
        return "result"
    if normalized_key in VALIDATION_COLUMN_ALIASES:
        return "validation"
    for alias in RESULT_COLUMN_ALIASES:
        if normalized_key.endswith(f" {alias}") or normalized_key.startswith(f"{alias} "):
            return "result"
    for alias in VALIDATION_COLUMN_ALIASES:
        if normalized_key.endswith(f" {alias}") or normalized_key.startswith(f"{alias} "):
            return "validation"
    return "other"


def _property_label(row: dict[str, Any]) -> str:
    for key in _PROPERTY_LABEL_KEYS:
        value = _text(row.get(key))
        if value:
            return value
    return ""


def _is_excluded_tensile_label(normalized: str) -> bool:
    if any(marker in normalized for marker in _TENSILE_EXCLUDE_MARKERS):
        return True
    if re.search(r"\bRE\b", normalized) and "TENSILE" not in normalized:
        return True
    return False


def _matches_elongation(normalized: str) -> bool:
    if "ELONGATION" in normalized:
        return True
    if "ELONGATION AFTER FRACTURE" in normalized:
        return True
    if normalized in {"A%", "A %"} or re.fullmatch(r"A\s*%?", normalized):
        return True
    if re.search(r"\bA\s*%\b", normalized):
        return True
    return False


def _matches_tensile(normalized: str) -> bool:
    if _is_excluded_tensile_label(normalized):
        return False
    if "TENSILE STRENGTH" in normalized:
        return True
    if "ULTIMATE TENSILE" in normalized:
        return True
    if re.search(r"\bRM\b", normalized):
        return True
    return False


def _matches_yield_family(normalized: str) -> str | None:
    if "RP0.2" in normalized or re.search(r"\bRP\s*0[\.,]?\s*2\b", normalized):
        return "yield_rp02"
    if "RP1.0" in normalized or re.search(r"\bRP\s*1[\.,]?\s*0\b", normalized):
        return "yield_rp10"
    if "PROOF STRENGTH" in normalized or "YIELD STRENGTH" in normalized:
        return "yield"
    if re.search(r"\bRE\b", normalized):
        return "yield"
    return None


def classify_property_label(label: str) -> str | None:
    normalized = normalize_header(label)
    if not normalized:
        return None
    if _matches_elongation(normalized):
        return "elongation"
    if _matches_tensile(normalized):
        return "tensile"
    return _matches_yield_family(normalized)


def _parse_strength_value(raw: Any, *, kind: str) -> float | None:
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return None
    return parse_numeric(raw, kind=kind).value  # type: ignore[arg-type]


def _parse_elongation_value(raw: Any) -> tuple[float | None, str | None]:
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return None, None
    text = _text(raw)
    if "/" in text:
        primary = text.split("/", 1)[0].strip()
        value = parse_numeric(primary, kind="elongation_pct").value
        return value, text
    value = parse_numeric(raw, kind="elongation_pct").value
    return value, None


def _coerce_numeric_values(raw: Any) -> list[float]:
    if isinstance(raw, (list, tuple)):
        values: list[float] = []
        for item in raw:
            values.extend(_coerce_numeric_values(item))
        return values
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return [float(raw)]
    if isinstance(raw, str):
        cleaned = raw.strip()
        if not cleaned:
            return []
        if "/" in cleaned and not any(ch.isalpha() for ch in cleaned.replace("/", "")):
            primary = cleaned.split("/", 1)[0].strip()
            parsed = parse_numeric(primary, kind="elongation_pct").value
            return [parsed] if parsed is not None else []
        parsed = parse_numeric(cleaned, kind="tensile_mpa").value
        if parsed is not None:
            return [parsed]
        parsed = parse_numeric(cleaned, kind="yield_mpa").value
        if parsed is not None:
            return [parsed]
    return []


def _extract_result_candidates(row: dict[str, Any]) -> tuple[list[tuple[str, Any]], list[tuple[str, Any]], list[tuple[str, Any]]]:
    result_entries: list[tuple[str, Any]] = []
    validation_entries: list[tuple[str, Any]] = []
    other_entries: list[tuple[str, Any]] = []

    for key, value in row.items():
        if value is None or (isinstance(value, str) and not value.strip()):
            continue
        normalized_key = _normalize_column_key(str(key))
        if normalized_key in _ROW_META_KEYS:
            continue
        kind = _column_kind(normalized_key)
        if kind == "result":
            result_entries.append((normalized_key, value))
        elif kind == "validation":
            validation_entries.append((normalized_key, value))
        else:
            other_entries.append((normalized_key, value))

    mechanical = row.get("mechanical_properties")
    if isinstance(mechanical, dict):
        for key, value in mechanical.items():
            if value is None:
                continue
            normalized_key = _normalize_column_key(str(key))
            if normalized_key in _ROW_META_KEYS:
                continue
            kind = _column_kind(normalized_key)
            if kind == "result":
                result_entries.append((f"mechanical_properties.{key}", value))
            elif kind == "validation":
                validation_entries.append((f"mechanical_properties.{key}", value))
            elif kind is None:
                other_entries.append((f"mechanical_properties.{key}", value))

    return result_entries, validation_entries, other_entries


def _select_result_value(
  result_entries: list[tuple[str, Any]],
  other_entries: list[tuple[str, Any]],
) -> tuple[Any | None, bool]:
    pool = result_entries or other_entries
    if not pool:
        return None, False

    numeric_by_source: dict[str, list[float]] = {}
    raw_by_source: dict[str, Any] = {}
    for source, raw in pool:
        numbers = _coerce_numeric_values(raw)
        if numbers:
            numeric_by_source[source] = numbers
        raw_by_source[source] = raw

    if not numeric_by_source and not raw_by_source:
        return None, False

    if result_entries:
        unique_values = {
            round(number, 6)
            for numbers in numeric_by_source.values()
            for number in numbers
        }
        if len(unique_values) > 1:
            return raw_by_source[result_entries[0][0]], True
        if len(result_entries) == 1:
            return result_entries[0][1], False
        first_source = result_entries[0][0]
        return raw_by_source.get(first_source, result_entries[0][1]), False

    if len(other_entries) == 1:
        return other_entries[0][1], False

    # Multiple unlabeled columns without an explicit Results header — uncertain.
    unique_values = {
        round(number, 6)
        for numbers in numeric_by_source.values()
        for number in numbers
    }
    if len(unique_values) > 1:
        return None, True
    return other_entries[0][1], len(other_entries) > 1


def map_mechanical_table_rows(table_rows: list[dict[str, Any]]) -> MechanicalTableMappingResult:
    """Map labeled mechanical table rows to canonical mechanical properties."""

    yield_rp02: float | None = None
    yield_rp10: float | None = None
    yield_generic: float | None = None
    tensile: float | None = None
    elongation: float | None = None
    elongation_raw: str | None = None

    uncertain = False
    row_traces: list[dict[str, Any]] = []
    field_candidates: dict[str, list[float]] = {
        "yield_strength_mpa": [],
        "tensile_strength_mpa": [],
        "elongation_percentage": [],
    }

    for index, row in enumerate(table_rows):
        if not isinstance(row, dict):
            continue
        label = _property_label(row)
        if not label:
            continue
        classification = classify_property_label(label)
        if classification is None:
            continue

        result_entries, validation_entries, other_entries = _extract_result_candidates(row)
        selected_raw, row_uncertain = _select_result_value(result_entries, other_entries)
        if row_uncertain:
            uncertain = True

        row_trace: dict[str, Any] = {
            "row_index": index,
            "property_label": label,
            "classification": classification,
            "result_columns": [source for source, _ in result_entries],
            "validation_columns": [source for source, _ in validation_entries],
            "selected_raw": selected_raw,
            "uncertain": row_uncertain,
        }
        row_traces.append(row_trace)

        if selected_raw is None and not result_entries and validation_entries:
            # Only specification/min/max columns present — never extract from them.
            row_trace["skipped"] = "validation_only_columns"
            continue

        if classification == "elongation":
            parsed, raw = _parse_elongation_value(selected_raw)
            if parsed is not None:
                field_candidates["elongation_percentage"].append(parsed)
                elongation = parsed
                if raw:
                    elongation_raw = raw
        elif classification == "tensile":
            parsed = _parse_strength_value(selected_raw, kind="tensile_mpa")
            if parsed is not None:
                field_candidates["tensile_strength_mpa"].append(parsed)
                tensile = parsed
        elif classification == "yield_rp02":
            parsed = _parse_strength_value(selected_raw, kind="yield_mpa")
            if parsed is not None:
                field_candidates["yield_strength_mpa"].append(parsed)
                yield_rp02 = parsed
        elif classification == "yield_rp10":
            parsed = _parse_strength_value(selected_raw, kind="yield_mpa")
            if parsed is not None:
                yield_rp10 = parsed
        elif classification == "yield":
            parsed = _parse_strength_value(selected_raw, kind="yield_mpa")
            if parsed is not None:
                field_candidates["yield_strength_mpa"].append(parsed)
                yield_generic = parsed

    canonical_yield = yield_rp02 if yield_rp02 is not None else yield_generic
    if yield_rp02 is not None and yield_rp10 is not None and yield_rp02 != yield_rp10:
        pass  # expected dual proof rows; Rp0.2 remains canonical

    for field_name, candidates in field_candidates.items():
        unique = {round(value, 6) for value in candidates}
        if len(unique) > 1:
            uncertain = True

    auxiliary: dict[str, Any] = {}
    if yield_rp10 is not None:
        auxiliary["yield_strength_rp1_0_mpa"] = yield_rp10
    if elongation_raw:
        auxiliary["elongation_raw"] = elongation_raw

    mechanical_properties = {
        "yield_strength_mpa": canonical_yield,
        "tensile_strength_mpa": tensile,
        "elongation_percentage": elongation,
    }

    tokens: list[str] = []
    if uncertain:
        tokens.append("mechanical_table_alignment_uncertain")

    return MechanicalTableMappingResult(
        mechanical_properties=mechanical_properties,
        auxiliary=auxiliary,
        uncertain=uncertain,
        tokens=tokens,
        trace={
            "strategy": "mechanical_table_row_column_alignment",
            "row_traces": row_traces,
            "field_candidates": field_candidates,
        },
    )


_ELEMENT_SYMBOL_ALIASES: dict[str, str] = {
    "C": "C",
    "CARBON": "C",
    "SI": "Si",
    "SILICON": "Si",
    "MN": "Mn",
    "MANGANESE": "Mn",
    "P": "P",
    "PHOSPHORUS": "P",
    "PHOSPHOROUS": "P",
    "S": "S",
    "SULPHUR": "S",
    "SULFUR": "S",
    "CU": "Cu",
    "COPPER": "Cu",
    "NI": "Ni",
    "NICKEL": "Ni",
    "CR": "Cr",
    "CHROMIUM": "Cr",
    "MO": "Mo",
    "MOLYBDENUM": "Mo",
    "V": "V",
    "VANADIUM": "V",
    "AL": "Al",
    "ALUMINIUM": "Al",
    "ALUMINUM": "Al",
    "NB": "Nb",
    "NIOBIUM": "Nb",
    "COLUMBIUM": "Nb",
    "TI": "Ti",
    "TITANIUM": "Ti",
    "N": "N",
    "NITROGEN": "N",
    "B": "B",
    "BORON": "B",
    "FE": "Fe",
    "IRON": "Fe",
    "W": "W",
    "TUNGSTEN": "W",
    "CO": "Co",
    "COBALT": "Co",
    "PB": "Pb",
    "LEAD": "Pb",
    "SN": "Sn",
    "TIN": "Sn",
    "AS": "As",
    "ARSENIC": "As",
    "SB": "Sb",
    "ANTIMONY": "Sb",
    "ZR": "Zr",
    "ZIRCONIUM": "Zr",
    "CA": "Ca",
    "CALCIUM": "Ca",
    "MG": "Mg",
    "MAGNESIUM": "Mg",
    "ZN": "Zn",
    "ZINC": "Zn",
}


@dataclass
class ChemicalTableMappingResult:
    chemical_composition: dict[str, float] = field(default_factory=dict)
    uncertain: bool = False
    tokens: list[str] = field(default_factory=list)
    trace: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chemical_composition": dict(self.chemical_composition),
            "uncertain": self.uncertain,
            "tokens": list(self.tokens),
            "trace": dict(self.trace),
        }


def _parse_chemistry_value(raw: Any) -> float | None:
    if raw is None or isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        value = float(raw)
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    text = _text(raw)
    if not text:
        return None
    cleaned = text.replace("%", "").replace(" ", "").replace(",", ".")
    try:
        value = float(cleaned)
    except ValueError:
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def parse_chemistry_value(raw: Any) -> float | None:
    return _parse_chemistry_value(raw)


def canonicalize_element_symbol(value: Any) -> str | None:
    """Return a canonical chemical-element symbol for a table label."""

    if value is None:
        return None

    normalized = re.sub(r"[^A-Z0-9]", "", str(value).upper())
    if not normalized:
        return None

    return _ELEMENT_SYMBOL_ALIASES.get(normalized)


def _chemical_property_label(row: dict[str, Any]) -> str:
    for key in (*_PROPERTY_LABEL_KEYS, "element"):
        value = _text(row.get(key))
        if value:
            return value
    return ""


def _row_looks_like_chemical_property(row: dict[str, Any]) -> bool:
    label = _chemical_property_label(row)
    if label and canonicalize_element_symbol(label):
        return True
    for key in row:
        if canonicalize_element_symbol(str(key)):
            return True
    return False


def collect_chemical_table_rows(
    item_payload: dict[str, Any],
    items_raw: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Collect explicit or derived chemical composition table rows from Stage B output."""

    explicit = item_payload.get("chemical_table_rows")
    if isinstance(explicit, list):
        rows = [row for row in explicit if isinstance(row, dict)]
        if rows:
            return rows

    derived: list[dict[str, Any]] = []
    for row in items_raw:
        if isinstance(row, dict) and _row_looks_like_chemical_property(row):
            derived.append(row)
    if len(derived) >= 2:
        return derived
    return []


def _first_source_page(*row_groups: list[dict[str, Any]]) -> int | None:
    for group in row_groups:
        for row in group:
            page = row.get("source_page")
            if isinstance(page, int) and not isinstance(page, bool) and page >= 1:
                return page
    return None


def map_chemical_table_rows(table_rows: list[dict[str, Any]]) -> ChemicalTableMappingResult:
    """Map labeled chemistry table rows to observed element values."""

    composition: dict[str, float] = {}
    uncertain = False
    row_traces: list[dict[str, Any]] = []
    field_candidates: dict[str, list[float]] = {}

    for index, row in enumerate(table_rows):
        if not isinstance(row, dict):
            continue
        label = _chemical_property_label(row)
        symbol = canonicalize_element_symbol(label) if label else None
        result_entries, validation_entries, other_entries = _extract_result_candidates(row)
        selected_raw, row_uncertain = _select_result_value(result_entries, other_entries)
        if row_uncertain:
            uncertain = True

        if symbol is not None:
            row_trace: dict[str, Any] = {
                "row_index": index,
                "property_label": label,
                "element": symbol,
                "result_columns": [source for source, _ in result_entries],
                "validation_columns": [source for source, _ in validation_entries],
                "selected_raw": selected_raw,
                "uncertain": row_uncertain,
            }
            row_traces.append(row_trace)
            if selected_raw is None and not result_entries and validation_entries:
                row_trace["skipped"] = "validation_only_columns"
                continue
            parsed = _parse_chemistry_value(selected_raw)
            if parsed is not None:
                composition[symbol] = parsed
                field_candidates.setdefault(symbol, []).append(parsed)
            continue

        wide_hits = 0
        for key, value in row.items():
            element = canonicalize_element_symbol(str(key))
            if element is None or value is None or (isinstance(value, str) and not str(value).strip()):
                continue
            normalized_key = _normalize_column_key(str(key))
            kind = _column_kind(normalized_key)
            if kind == "validation":
                continue
            parsed = _parse_chemistry_value(value)
            if parsed is None:
                continue
            composition[element] = parsed
            field_candidates.setdefault(element, []).append(parsed)
            wide_hits += 1
        if wide_hits:
            row_traces.append(
                {
                    "row_index": index,
                    "property_label": label,
                    "wide_element_columns": wide_hits,
                    "uncertain": row_uncertain,
                }
            )

    for candidates in field_candidates.values():
        unique = {round(value, 6) for value in candidates}
        if len(unique) > 1:
            uncertain = True

    tokens: list[str] = []
    if uncertain:
        tokens.append("chemical_table_alignment_uncertain")

    return ChemicalTableMappingResult(
        chemical_composition=composition,
        uncertain=uncertain,
        tokens=tokens,
        trace={
            "strategy": "chemical_table_row_column_alignment",
            "row_traces": row_traces,
            "field_candidates": field_candidates,
        },
    )


def _row_looks_like_mechanical_property(row: dict[str, Any]) -> bool:
    label = _property_label(row)
    if not label:
        return False
    return classify_property_label(label) is not None


def collect_mechanical_table_rows(
    item_payload: dict[str, Any],
    items_raw: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Collect explicit or derived mechanical table rows from Stage B output."""

    explicit = item_payload.get("mechanical_table_rows")
    if isinstance(explicit, list):
        rows = [row for row in explicit if isinstance(row, dict)]
        if rows:
            return rows

    derived: list[dict[str, Any]] = []
    for row in items_raw:
        if isinstance(row, dict) and _row_looks_like_mechanical_property(row):
            derived.append(row)
    if len(derived) >= 2:
        return derived
    return []


def apply_mechanical_table_mapping(
    rows: list[dict[str, Any]],
    *,
    item_payload: dict[str, Any],
    items_raw: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str], dict[str, Any]]:
    """Apply mechanical and chemical table mapping to extracted item rows in-place."""

    table_rows = collect_mechanical_table_rows(item_payload, items_raw)
    mapping = map_mechanical_table_rows(table_rows) if table_rows else MechanicalTableMappingResult()
    chemical_rows = collect_chemical_table_rows(item_payload, items_raw)
    chemical = map_chemical_table_rows(chemical_rows) if chemical_rows else ChemicalTableMappingResult()

    tokens = list(mapping.tokens)
    tokens.extend(chemical.tokens)
    trace: dict[str, Any] = dict(mapping.trace) if mapping.trace else {}
    if chemical.trace or chemical.chemical_composition:
        trace["chemical"] = chemical.to_dict()

    has_mechanicals = any(value is not None for value in mapping.mechanical_properties.values())
    has_chemistry = bool(chemical.chemical_composition)
    source_page = _first_source_page(table_rows, chemical_rows, items_raw)

    if not rows:
        if not has_mechanicals and not has_chemistry:
            return rows, tokens, trace
        synthesized: dict[str, Any] = {
            "needs_review": mapping.uncertain or chemical.uncertain,
        }
        if has_mechanicals:
            synthesized["mechanical_properties"] = dict(mapping.mechanical_properties)
            if mapping.auxiliary:
                synthesized["_mechanical_auxiliary"] = mapping.auxiliary
            mapping.trace["synthesized_item_from_mechanical_table"] = True
            trace["synthesized_item_from_mechanical_table"] = True
        if has_chemistry:
            synthesized["chemical_composition"] = dict(chemical.chemical_composition)
            trace["synthesized_item_from_chemical_table"] = True
        if source_page is not None:
            synthesized["source_page"] = source_page
        return [synthesized], tokens, trace

    target = rows[0]
    if has_mechanicals:
        mechanical = target.get("mechanical_properties")
        if not isinstance(mechanical, dict):
            mechanical = {}
            target["mechanical_properties"] = mechanical
        for field_name, value in mapping.mechanical_properties.items():
            if value is not None:
                mechanical[field_name] = value
        if mapping.auxiliary:
            target["_mechanical_auxiliary"] = mapping.auxiliary

    if has_chemistry:
        existing = target.get("chemical_composition")
        merged = dict(existing) if isinstance(existing, dict) else {}
        for element, value in chemical.chemical_composition.items():
            if merged.get(element) is None:
                merged[element] = value
        target["chemical_composition"] = merged

    if source_page is not None and target.get("source_page") is None:
        target["source_page"] = source_page

    if mapping.uncertain or chemical.uncertain:
        target["needs_review"] = True

    return rows, tokens, trace


__all__ = [
    "ChemicalTableMappingResult",
    "MechanicalTableMappingResult",
    "apply_mechanical_table_mapping",
    "canonicalize_element_symbol",
    "classify_property_label",
    "collect_chemical_table_rows",
    "collect_mechanical_table_rows",
    "map_chemical_table_rows",
    "map_mechanical_table_rows",
    "parse_chemistry_value",
]
