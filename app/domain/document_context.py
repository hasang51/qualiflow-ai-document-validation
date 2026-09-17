"""Generic document context builder + conservative row propagation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.domain.grade_alias_registry import GradeSemanticResolution, resolve_grade_semantics

PRIMARY_KEY_FIELDS: tuple[str, ...] = ("heat_number", "item_id", "serial_number", "coil_id", "batch_id")


@dataclass
class DocumentContext:
    header_grade: str | None = None
    header_grade_resolution: GradeSemanticResolution | None = None
    supplier_name: str | None = None
    certificate_metadata: dict[str, Any] = field(default_factory=dict)
    table_headers: list[str] = field(default_factory=list)
    section_headers: list[str] = field(default_factory=list)
    identifier_index: dict[str, dict[str, Any]] = field(default_factory=dict)
    propagation_notes: list[dict[str, Any]] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "header_grade": self.header_grade,
            "header_grade_resolution": (
                self.header_grade_resolution.to_dict() if self.header_grade_resolution else None
            ),
            "supplier_name": self.supplier_name,
            "certificate_metadata": dict(self.certificate_metadata),
            "table_headers": list(self.table_headers),
            "section_headers": list(self.section_headers),
            "identifier_index": dict(self.identifier_index),
            "propagation_notes": list(self.propagation_notes),
            "conflicts": list(self.conflicts),
        }


def _extract_header_grade(metadata: dict[str, Any]) -> str | None:
    for key in ("header_grade", "document_grade", "default_grade_hint", "grade"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def build_document_context(
    rows: list[dict[str, Any]],
    *,
    metadata: dict[str, Any] | None = None,
    pages_meta: list[dict[str, Any]] | None = None,
) -> DocumentContext:
    meta = metadata or {}
    context = DocumentContext(
        supplier_name=str(meta.get("supplier_name")) if meta.get("supplier_name") else None,
        certificate_metadata={
            key: meta.get(key)
            for key in (
                "document_type",
                "certificate_date",
                "batch_number",
                "lot_number",
                "colata_number",
                "certificate_number",
                "certificate_no",
                "order_number",
                "order_no",
                "product_description",
            )
            if meta.get(key) is not None
        },
    )
    header_grade = _extract_header_grade(meta)
    context.header_grade = header_grade
    if header_grade:
        context.header_grade_resolution = resolve_grade_semantics(header_grade)

    for row_index, row in enumerate(rows):
        for field in PRIMARY_KEY_FIELDS:
            value = row.get(field)
            if isinstance(value, str) and value.strip():
                bucket = context.identifier_index.setdefault(value.strip(), {"rows": [], "fields": set()})
                bucket["rows"].append(row_index)
                bucket["fields"].add(field)
    # Ensure JSON-serialisable sets.
    for value in context.identifier_index.values():
        value["fields"] = sorted(value["fields"])

    if pages_meta:
        for page in pages_meta:
            table_parsing = page.get("table_parsing") if isinstance(page, dict) else None
            if isinstance(table_parsing, dict):
                headers = table_parsing.get("headers")
                if isinstance(headers, list):
                    context.table_headers.extend(str(h) for h in headers if h)
            variant = page.get("variant_selection") if isinstance(page, dict) else None
            if isinstance(variant, dict):
                reasons = variant.get("selection_reasoning")
                if isinstance(reasons, list):
                    context.section_headers.extend(str(reason) for reason in reasons if reason)
    context.table_headers = sorted(set(context.table_headers))
    context.section_headers = sorted(set(context.section_headers))
    return context


def propagate_context_to_rows(rows: list[dict[str, Any]], context: DocumentContext) -> list[str]:
    """Apply conservative propagation in place and return structured tokens."""

    tokens: list[str] = []
    header_grade_resolution = context.header_grade_resolution
    if (
        header_grade_resolution is not None
        and not header_grade_resolution.unresolved
        and header_grade_resolution.canonical
    ):
        for row_index, row in enumerate(rows):
            row_grade = row.get("grade")
            if isinstance(row_grade, str) and row_grade.strip():
                row_grade_res = resolve_grade_semantics(row_grade)
                if (
                    row_grade_res.canonical
                    and row_grade_res.canonical != header_grade_resolution.canonical
                ):
                    conflict = f"header_row_conflict:grade:row{row_index}"
                    tokens.append(conflict)
                    context.conflicts.append(conflict)
                continue
            row["grade"] = header_grade_resolution.canonical
            row.setdefault("grade_provenance", "header_context")
            note = {
                "row_index": row_index,
                "field": "grade",
                "source": "document_header",
                "value": header_grade_resolution.canonical,
                "reason": "row value missing and header grade resolved confidently",
            }
            context.propagation_notes.append(note)
            tokens.append("context_propagation:grade_from_header")
    return sorted(set(tokens))


__all__ = ["DocumentContext", "build_document_context", "propagate_context_to_rows"]
