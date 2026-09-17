"""Semantic normalization for row fields and grade/spec tokens."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.domain.field_synonym_registry import FieldResolution, resolve_field_name
from app.domain.grade_alias_registry import GradeSemanticResolution, resolve_grade_semantics
from app.domain.grade_registry import is_explicit_grade_field_label


@dataclass
class RowSemanticNormalization:
    row_index: int
    canonical_values: dict[str, Any]
    field_resolutions: dict[str, FieldResolution]
    grade_resolution: GradeSemanticResolution | None = None
    unresolved_fields: list[str] = field(default_factory=list)
    ambiguous_fields: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "row_index": self.row_index,
            "canonical_values": dict(self.canonical_values),
            "field_resolutions": {
                key: value.to_dict() for key, value in self.field_resolutions.items()
            },
            "grade_resolution": self.grade_resolution.to_dict() if self.grade_resolution else None,
            "unresolved_fields": list(self.unresolved_fields),
            "ambiguous_fields": list(self.ambiguous_fields),
        }


def _explicit_grade_source_label(row: dict[str, Any]) -> str | None:
    for key, value in row.items():
        if value is None or (isinstance(value, str) and not value.strip()):
            continue
        resolution = resolve_field_name(str(key))
        if resolution.canonical_field != "grade":
            continue
        if key == "grade" or is_explicit_grade_field_label(str(key)):
            return str(key)
    return None


def _canonicalize_row_keys(row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, FieldResolution], list[str]]:
    canonical: dict[str, Any] = {}
    resolutions: dict[str, FieldResolution] = {}
    unresolved: list[str] = []
    for key, value in row.items():
        resolution = resolve_field_name(str(key))
        resolutions[str(key)] = resolution
        target = resolution.canonical_field or str(key)
        if resolution.canonical_field is None:
            unresolved.append(str(key))
        existing = canonical.get(target)
        if (
            target != str(key)
            and existing not in (None, "", [])
            and value not in (None, "", [])
        ):
            continue
        canonical[target] = value
    return canonical, resolutions, unresolved


def normalize_rows_semantics(rows: list[dict[str, Any]]) -> list[RowSemanticNormalization]:
    """Normalize row keys/grade strings into canonical semantic structures."""

    normalized_rows: list[RowSemanticNormalization] = []
    for row_index, row in enumerate(rows):
        canonical_values, field_resolutions, unresolved = _canonicalize_row_keys(row)
        ambiguous_fields: list[str] = []
        grade_resolution: GradeSemanticResolution | None = None
        grade_source = _explicit_grade_source_label(row)
        if grade_source:
            canonical_values["grade_field_label"] = grade_source
            canonical_values.setdefault("grade_provenance", "labeled_field")
        if "grade" in canonical_values:
            grade_resolution = resolve_grade_semantics(canonical_values.get("grade"))
            if grade_resolution.ambiguous:
                ambiguous_fields.append("grade")
            canonical_values["grade_candidates"] = list(grade_resolution.candidates)
            canonical_values["grade_tokens"] = list(grade_resolution.tokens)
        normalized_rows.append(
            RowSemanticNormalization(
                row_index=row_index,
                canonical_values=canonical_values,
                field_resolutions=field_resolutions,
                grade_resolution=grade_resolution,
                unresolved_fields=unresolved,
                ambiguous_fields=ambiguous_fields,
            )
        )
    return normalized_rows

