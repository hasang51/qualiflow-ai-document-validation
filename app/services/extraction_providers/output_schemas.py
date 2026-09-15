from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class _ForbidModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LabeledDate(_ForbidModel):
    label: str
    value: str


class StageAFieldConfidence(_ForbidModel):
    heat_number: float | None = None
    batch_number: float | None = None
    certificate_number: float | None = None
    order_number: float | None = None


class StageAMetadata(_ForbidModel):
    supplier_name: str
    document_type: str
    product_category: Literal["WIRE_ROPE", "STRUCTURAL_STEEL", "PIPE", "OTHER"] | None = None
    certificate_date: str | None = None
    labeled_dates: list[LabeledDate] | None = None
    heat_number: str | None = None
    batch_number: str | None = None
    certificate_number: str | None = None
    order_number: str | None = None
    header_grade: str | None = None
    product_description: str | None = None
    weight_or_length: str | None = None
    ai_analysis_remarks: str | None = None
    confidence_score: float
    field_confidence: StageAFieldConfidence | None = None


class StageBMechanicalProperties(_ForbidModel):
    yield_strength_mpa: float | None = None
    tensile_strength_mpa: float | None = None
    elongation_percentage: float | None = None


class StageBFieldConfidence(_ForbidModel):
    heat_number: float | None = None
    batch_number: float | None = None
    item_id: float | None = None
    certificate_number: float | None = None
    order_number: float | None = None
    yield_strength_mpa: float | None = None
    tensile_strength_mpa: float | None = None
    elongation_percentage: float | None = None


class StageBItem(_ForbidModel):
    item_id: str | None = None
    heat_number: str | None = None
    batch_number: str | None = None
    certificate_number: str | None = None
    order_number: str | None = None
    grade: str | None = None
    weight_or_length: str | None = None
    mechanical_properties: StageBMechanicalProperties | None = None
    row_confidence: float | None = None
    needs_review: bool | None = None
    field_confidence: StageBFieldConfidence | None = None
    source_page: int | None = None


class StageBLineItems(_ForbidModel):
    extraction_audit: str | None = None
    total_items_detected: int
    mechanical_table_rows: list[dict[str, Any]] | None = Field(
        default=None,
        description="Property-per-row mechanical tables; extra keys are allowed.",
    )
    items: list[StageBItem]


def response_format_for_model(model: type[BaseModel], name: str, *, strict: bool) -> dict[str, Any]:
    """Build a Mantle Chat Completions json_schema response_format payload."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "strict": strict,
            "schema": model.model_json_schema(),
        },
    }
