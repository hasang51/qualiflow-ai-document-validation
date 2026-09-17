from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from app.domain.grade_registry import resolve_grade
from app.services.identity_field_mapper import (
    HEADER_DIMENSION_KEYS,
    HEADER_GRADE_KEYS,
    HEADER_PRODUCT_KEYS,
    extract_dimension_fragment,
    is_standards_only,
    looks_like_classification_token,
)


PROPERTY_LABELS: dict[str, tuple[str, ...]] = {
    "yield_strength_mpa": ("yield", "snervamento", "proof"),
    "tensile_strength_mpa": ("tensile", "carico di rottura", "rm"),
    "elongation_percentage": ("elongation", "allungamento", "a5"),
}
HEADER_HEAT_KEYS = ("heat_number", "header_heat_number")
HEADER_BATCH_KEYS = ("batch_number",)
HEADER_LOT_KEYS = ("lot_number",)
HEADER_COLATA_KEYS = ("colata_number",)
HEADER_CAST_KEYS = ("cast_number",)
HEADER_CERT_KEYS = ("certificate_number",)
HEADER_ORDER_KEYS = ("order_number",)
HEADER_WEIGHT_KEYS = ("weight_or_length", "product_weight", "quantity")
IDENTITY_PRESERVE_KEYS = (
    "product_name",
    "grade",
    "dimensions",
    "standards",
    "chemical_composition",
    "certificate_number",
    "order_number",
    "batch_number",
    "lot_number",
    "colata_number",
    "heat_number",
    "weight_or_length",
)


@dataclass
class RowShapeNormalizationResult:
    rows: list[dict[str, Any]]
    tokens: list[str] = field(default_factory=list)
    trace: dict[str, Any] = field(default_factory=dict)


def _text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _metadata_value(metadata: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _property_field(row: dict[str, Any]) -> str | None:
    label = f"{_text(row.get('item_id'))} {_text(row.get('grade'))}".lower()
    mechanical = row.get("mechanical_properties")
    if not isinstance(mechanical, dict):
        return None
    populated = [key for key, value in mechanical.items() if value is not None]
    if len(populated) == 1:
        return populated[0]
    for field_name, needles in PROPERTY_LABELS.items():
        if any(needle in label for needle in needles):
            return field_name
    return None


def _group_key(row: dict[str, Any]) -> str:
    grade = _text(row.get("grade"))
    if grade:
        return grade
    item_id = _text(row.get("item_id"))
    if "-" in item_id:
        return item_id.split("-", 1)[0].strip()
    return "default"


def _product_detail_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    for row in rows:
        item_id = _text(row.get("item_id")).lower()
        mechanical = row.get("mechanical_properties")
        if "product" in item_id and not isinstance(mechanical, dict):
            return row
    return None


def _confidence(values: list[Any]) -> float | None:
    numbers = [float(value) for value in values if isinstance(value, (int, float))]
    if not numbers:
        return None
    return max(0.05, min(1.0, sum(numbers) / len(numbers)))


def _has_complete_mechanicals(row: dict[str, Any]) -> bool:
    mechanical = row.get("mechanical_properties")
    if not isinstance(mechanical, dict):
        return False
    return all(
        mechanical.get(field) is not None
        for field in ("yield_strength_mpa", "tensile_strength_mpa", "elongation_percentage")
    )


def _looks_like_classification_id(value: Any) -> bool:
    return looks_like_classification_token(value)


def _first_present(rows: list[dict[str, Any]], key: str) -> Any:
    for row in rows:
        value = row.get(key)
        if value not in (None, "", []):
            return value
    return None


def _merge_standard_values(*groups: Any) -> list[str] | None:
    seen: set[str] = set()
    merged: list[str] = []
    for group in groups:
        values: list[str]
        if isinstance(group, list):
            values = [str(item).strip() for item in group if str(item).strip()]
        elif isinstance(group, str) and group.strip():
            values = [group.strip()]
        else:
            values = []
        for value in values:
            key = value.casefold()
            if key in seen:
                continue
            seen.add(key)
            merged.append(value)
    return merged or None


def _classification_label(row: dict[str, Any]) -> str | None:
    for key in ("item_id", "grade"):
        value = _text(row.get(key))
        if _looks_like_classification_id(value):
            return value
    return None


def _grade_hint(metadata: dict[str, Any]) -> str | None:
    explicit = _metadata_value(metadata, HEADER_GRADE_KEYS)
    if explicit and not looks_like_classification_token(explicit) and not is_standards_only(explicit):
        return explicit
    product = _metadata_value(metadata, HEADER_PRODUCT_KEYS)
    if not product:
        return None
    resolution = resolve_grade(product)
    if resolution.status in {"resolved", "resolved_dual"} and resolution.canonical:
        return resolution.canonical
    return None


def collapse_vertical_mechanical_rows(
    rows: list[dict[str, Any]],
    *,
    metadata: dict[str, Any],
) -> RowShapeNormalizationResult:
    """Collapse property-per-row LLM output into item-centric rows.

    Some MTCs report mechanical values vertically: one line for yield, one for
    tensile, one for elongation. The API schema, however, expects one material
    item row containing all mechanical properties. This normalizer only acts on
    clear vertical-table signatures and otherwise returns rows unchanged.
    """

    property_rows: list[tuple[int, dict[str, Any], str]] = []
    for index, row in enumerate(rows):
        field_name = _property_field(row)
        if field_name is not None:
            property_rows.append((index, row, field_name))

    if len(property_rows) < 2:
        return RowShapeNormalizationResult(rows=rows)
    if any(_text(row.get("heat_number")) for _, row, _ in property_rows):
        return RowShapeNormalizationResult(rows=rows)

    grouped: dict[str, dict[str, Any]] = {}
    for _, row, field_name in property_rows:
        group = grouped.setdefault(
            _group_key(row),
            {
                "mechanical_properties": {
                    "yield_strength_mpa": None,
                    "tensile_strength_mpa": None,
                    "elongation_percentage": None,
                },
                "row_confidences": [],
                "source_rows": [],
            },
        )
        mechanical = row.get("mechanical_properties") or {}
        value = mechanical.get(field_name) if isinstance(mechanical, dict) else None
        if value is not None and group["mechanical_properties"][field_name] is None:
            group["mechanical_properties"][field_name] = value
        group["row_confidences"].append(row.get("row_confidence"))
        group["source_rows"].append(row.get("item_id"))

    complete_groups = [
        (group_name, payload)
        for group_name, payload in grouped.items()
        if all(value is not None for value in payload["mechanical_properties"].values())
    ]
    if not complete_groups:
        return RowShapeNormalizationResult(rows=rows)

    selected_group, payload = complete_groups[0]
    product_row = _product_detail_row(rows)
    sources = [product_row, *rows] if product_row is not None else list(rows)
    heat_number = _metadata_value(metadata, HEADER_HEAT_KEYS) or _first_present(sources, "heat_number")
    grade_hint = _grade_hint(metadata)
    grade = grade_hint or _first_present(sources, "grade")
    if grade and (_looks_like_classification_id(grade) or is_standards_only(grade)):
        grade = grade_hint
    weight = _metadata_value(metadata, HEADER_WEIGHT_KEYS) or _first_present(sources, "weight_or_length")
    dimensions = _metadata_value(metadata, HEADER_DIMENSION_KEYS) or _first_present(sources, "dimensions")
    if not dimensions and weight:
        dimensions = extract_dimension_fragment(weight)
    product_name = _metadata_value(metadata, HEADER_PRODUCT_KEYS) or _first_present(sources, "product_name")
    classification_labels = [label for row in rows if (label := _classification_label(row))]
    standards = _merge_standard_values(
        _first_present(sources, "standards"),
        metadata.get("standards"),
        classification_labels,
    )

    collapsed = {
        "item_id": None,
        "heat_number": heat_number,
        "product_name": product_name,
        "grade": grade,
        "weight_or_length": weight,
        "dimensions": dimensions,
        "standards": standards,
        "mechanical_properties": payload["mechanical_properties"],
        "row_confidence": _confidence(payload["row_confidences"]),
    }
    for key in IDENTITY_PRESERVE_KEYS:
        if collapsed.get(key) not in (None, "", []):
            continue
        value = _first_present(sources, key)
        if key == "grade" and value and (_looks_like_classification_id(value) or is_standards_only(value)):
            continue
        if key == "product_name" and value and is_standards_only(value):
            continue
        collapsed[key] = value
    # item_id intentionally None: collapsed rows originate from vertically split
    # mechanical-property lines whose ``item_id`` cells are property labels —
    # not raster Item ID traceability columns.

    trace = {
        "strategy": "vertical_mechanical_table_collapse",
        "selected_group": selected_group,
        "source_rows": payload["source_rows"],
        "group_count": len(grouped),
        "product_detail_found": product_row is not None,
        "metadata_fields_used": {
            "heat_number": heat_number is not None,
            "grade": grade is not None,
            "weight_or_length": weight is not None,
            "product_name": product_name is not None,
            "dimensions": dimensions is not None,
        },
    }
    return RowShapeNormalizationResult(
        rows=[collapsed],
        tokens=["row_shape:vertical_mechanical_table_collapsed"],
        trace=trace,
    )


def _looks_like_heat_number(value: str) -> bool:
    canonical = re.sub(r"[^A-Z0-9]", "", value.upper())
    return bool(canonical) and any(ch.isdigit() for ch in canonical) and len(canonical) >= 4


def _copy_metadata_confidence(row: dict[str, Any], metadata: dict[str, Any], field_name: str) -> None:
    meta_conf = metadata.get("field_confidence")
    if not isinstance(meta_conf, dict):
        return
    confidence = meta_conf.get(field_name)
    if not isinstance(confidence, (int, float)):
        return
    row_conf = row.get("_identifier_confidence")
    if not isinstance(row_conf, dict):
        row_conf = {}
        row["_identifier_confidence"] = row_conf
    if row_conf.get(field_name) is None:
        row_conf[field_name] = float(confidence)


def backfill_single_item_context(
    rows: list[dict[str, Any]],
    *,
    metadata: dict[str, Any],
) -> RowShapeNormalizationResult:
    """Backfill obvious document-level identifiers into one extracted item."""

    if len(rows) != 1:
        return RowShapeNormalizationResult(rows=rows)
    row = dict(rows[0])
    tokens: list[str] = []
    heat_number = _metadata_value(metadata, HEADER_HEAT_KEYS)
    if not row.get("heat_number") and heat_number and _looks_like_heat_number(heat_number):
        row["heat_number"] = heat_number
        _copy_metadata_confidence(row, metadata, "heat_number")
        tokens.append("context_propagation:heat_number_from_metadata")
    batch_number = _metadata_value(metadata, HEADER_BATCH_KEYS)
    if not row.get("batch_number") and batch_number and _looks_like_heat_number(batch_number):
        row["batch_number"] = batch_number
        _copy_metadata_confidence(row, metadata, "batch_number")
        tokens.append("context_propagation:batch_number_from_metadata")
    lot_number = _metadata_value(metadata, HEADER_LOT_KEYS)
    if not row.get("lot_number") and lot_number and _looks_like_heat_number(lot_number):
        row["lot_number"] = lot_number
        _copy_metadata_confidence(row, metadata, "lot_number")
        tokens.append("context_propagation:lot_number_from_metadata")
    colata_number = _metadata_value(metadata, HEADER_COLATA_KEYS)
    if not row.get("colata_number") and colata_number and _looks_like_heat_number(colata_number):
        row["colata_number"] = colata_number
        _copy_metadata_confidence(row, metadata, "colata_number")
        tokens.append("context_propagation:colata_number_from_metadata")
    cast_number = _metadata_value(metadata, HEADER_CAST_KEYS)
    if not row.get("cast_number") and cast_number and _looks_like_heat_number(cast_number):
        row["cast_number"] = cast_number
        _copy_metadata_confidence(row, metadata, "cast_number")
        tokens.append("context_propagation:cast_number_from_metadata")
    certificate_number = _metadata_value(metadata, HEADER_CERT_KEYS)
    if not row.get("certificate_number") and certificate_number:
        row["certificate_number"] = certificate_number
        _copy_metadata_confidence(row, metadata, "certificate_number")
        tokens.append("context_propagation:certificate_number_from_metadata")
    order_number = _metadata_value(metadata, HEADER_ORDER_KEYS)
    if not row.get("order_number") and order_number:
        row["order_number"] = order_number
        _copy_metadata_confidence(row, metadata, "order_number")
        tokens.append("context_propagation:order_number_from_metadata")
    grade = _metadata_value(metadata, HEADER_GRADE_KEYS)
    if (
        not row.get("grade")
        and grade
        and not looks_like_classification_token(grade)
        and not is_standards_only(grade)
    ):
        row["grade"] = grade
        tokens.append("context_propagation:grade_from_metadata")
    product_name = _metadata_value(metadata, HEADER_PRODUCT_KEYS)
    if not row.get("product_name") and product_name and not is_standards_only(product_name):
        row["product_name"] = product_name
        tokens.append("context_propagation:product_name_from_metadata")
    dimensions = _metadata_value(metadata, HEADER_DIMENSION_KEYS)
    if not row.get("dimensions") and dimensions:
        row["dimensions"] = dimensions
        tokens.append("context_propagation:dimensions_from_metadata")
    elif not row.get("dimensions") and row.get("weight_or_length"):
        extracted = extract_dimension_fragment(row.get("weight_or_length"))
        if extracted:
            row["dimensions"] = extracted
            tokens.append("context_propagation:dimensions_from_weight")
    header_standards = metadata.get("standards")
    if not row.get("standards") and header_standards:
        row["standards"] = _merge_standard_values(header_standards)
        tokens.append("context_propagation:standards_from_metadata")
    weight = _metadata_value(metadata, HEADER_WEIGHT_KEYS)
    if not row.get("weight_or_length") and weight:
        row["weight_or_length"] = weight
        tokens.append("context_propagation:weight_from_metadata")
    if not tokens:
        return RowShapeNormalizationResult(rows=rows)
    return RowShapeNormalizationResult(rows=[row], tokens=tokens, trace={"strategy": "single_item_metadata_backfill"})


def collapse_alternative_classification_rows(
    rows: list[dict[str, Any]],
    *,
    metadata: dict[str, Any],
) -> RowShapeNormalizationResult:
    """Collapse alternate classification rows for one product into one item.

    Some certificates show multiple classification rows (for example M21 and
    C1) for the same welding-wire product. When those rows share one propagated
    product grade and have no row-level heat/weight identifiers, they are
    alternate standards, not separate material items. Keep the first complete
    classification row as the item evidence and retain the others in trace.
    """

    if len(rows) < 2 or len(rows) > 4:
        return RowShapeNormalizationResult(rows=rows)
    heat_values = {_text(row.get("heat_number")) for row in rows if _text(row.get("heat_number"))}
    if len(heat_values) > 1 or any(_text(row.get("weight_or_length")) for row in rows):
        return RowShapeNormalizationResult(rows=rows)
    labels = [_classification_label(row) for row in rows]
    if not all(_has_complete_mechanicals(row) and label for row, label in zip(rows, labels)):
        return RowShapeNormalizationResult(rows=rows)
    grade = _grade_hint(metadata)
    if grade and _looks_like_classification_id(grade):
        grade = None
    if not grade:
        explicit_grades = {
            _text(row.get("grade"))
            for row in rows
            if _text(row.get("grade")) and not _looks_like_classification_id(row.get("grade"))
        }
        grade = next(iter(explicit_grades)) if len(explicit_grades) == 1 else None

    selected = dict(rows[0])
    selected["item_id"] = None
    selected["grade"] = grade
    selected["product_name"] = selected.get("product_name") or _metadata_value(metadata, HEADER_PRODUCT_KEYS)
    selected["dimensions"] = selected.get("dimensions") or _metadata_value(metadata, HEADER_DIMENSION_KEYS)
    selected["standards"] = _merge_standard_values(selected.get("standards"), labels, metadata.get("standards"))
    heat_number = _metadata_value(metadata, HEADER_HEAT_KEYS) or (next(iter(heat_values)) if heat_values else None)
    if heat_number:
        selected["heat_number"] = heat_number
    batch_number = _metadata_value(metadata, HEADER_BATCH_KEYS)
    if batch_number and not selected.get("batch_number"):
        selected["batch_number"] = batch_number
    lot_number = _metadata_value(metadata, HEADER_LOT_KEYS)
    if lot_number and not selected.get("lot_number"):
        selected["lot_number"] = lot_number
    colata_number = _metadata_value(metadata, HEADER_COLATA_KEYS)
    if colata_number and not selected.get("colata_number"):
        selected["colata_number"] = colata_number
    certificate_number = _metadata_value(metadata, HEADER_CERT_KEYS)
    if certificate_number and not selected.get("certificate_number"):
        selected["certificate_number"] = certificate_number
    order_number = _metadata_value(metadata, HEADER_ORDER_KEYS)
    if order_number and not selected.get("order_number"):
        selected["order_number"] = order_number
    weight = _metadata_value(metadata, HEADER_WEIGHT_KEYS)
    if weight:
        selected["weight_or_length"] = weight
    trace = {
        "strategy": "alternative_classification_rows_collapsed",
        "selected_source_label": labels[0],
        "discarded_alternative_labels": labels[1:],
        "resolved_grade": grade,
        "metadata_heat_used": heat_number is not None,
    }
    return RowShapeNormalizationResult(
        rows=[selected],
        tokens=["row_shape:alternative_classification_rows_collapsed"],
        trace=trace,
    )


__all__ = [
    "RowShapeNormalizationResult",
    "backfill_single_item_context",
    "collapse_alternative_classification_rows",
    "collapse_vertical_mechanical_rows",
]
