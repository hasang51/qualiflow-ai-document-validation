from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class MechanicalProperties(BaseModel):
    yield_strength_mpa: Optional[float] = Field(default=None)
    tensile_strength_mpa: Optional[float] = Field(default=None)
    elongation_percentage: Optional[float] = Field(default=None)


class ValidationResult(BaseModel):
    """Outcome of validating a single row.

    ``is_compliant`` is tri-state:

    - ``True``  => resolved & compliant,
    - ``False`` => resolved & non-compliant,
    - ``None``  => unresolved / not applicable / extraction uncertain.

    ``outcome`` is the narrow machine-readable tag the frontend uses to
    pick a badge.
    """

    is_compliant: Optional[bool] = Field(default=True)
    deviations: List[str] = Field(default_factory=list)
    outcome: Optional[str] = Field(default=None)
    rule_evidence: List[Dict[str, Any]] = Field(default_factory=list)


class ExtractedItem(BaseModel):
    item_id: Optional[str] = Field(default=None)
    pipe_id: Optional[str] = Field(default=None)
    heat_number: Optional[str] = Field(default=None)
    batch_number: Optional[str] = Field(default=None)
    lot_number: Optional[str] = Field(default=None)
    colata_number: Optional[str] = Field(default=None)
    cast_number: Optional[str] = Field(default=None)
    charge_number: Optional[str] = Field(default=None)
    coil_number: Optional[str] = Field(default=None)
    certificate_number: Optional[str] = Field(default=None)
    order_number: Optional[str] = Field(default=None)
    order_date: Optional[str] = Field(default=None)
    traceability_identifier_type: Optional[str] = Field(default=None)
    traceability_identifier_label: Optional[str] = Field(default=None)
    traceability_identifier_value: Optional[str] = Field(default=None)
    product_name: Optional[str] = Field(default=None)
    product_details: Optional[str] = Field(default=None)
    grade: Optional[str] = Field(default=None)
    weight_or_length: Optional[str] = Field(default=None)
    dimensions: Optional[str] = Field(default=None)
    standards: Optional[List[str]] = Field(default=None)
    classifications: Optional[List[str]] = Field(default=None)
    chemical_composition: Optional[Dict[str, Optional[float]]] = Field(default=None)
    mechanical_properties: Optional[MechanicalProperties] = Field(default=None)
    validation: Optional[ValidationResult] = Field(default=None)
    row_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    needs_review: bool = Field(default=False)
    # Phase 1 additions (all optional to keep API shape backward-compatible)
    grade_resolution: Optional[Dict[str, Any]] = Field(default=None)
    grade_provenance: Optional[str] = Field(default=None)
    grade_field_label: Optional[str] = Field(default=None)
    traceability_status: Optional[str] = Field(default=None)
    traceability_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    identifier_visibility_verified: Optional[bool] = Field(default=None)
    accepted_identifier_values: Dict[str, Any] = Field(default_factory=dict)
    raw_identifier_candidates: Dict[str, Any] = Field(default_factory=dict)
    source_page: Optional[int] = Field(default=None)


class UniversalDocumentExtraction(BaseModel):
    supplier_name: str = Field(...)
    document_type: str = Field(...)
    product_category: Optional[str] = Field(default=None)
    certificate_date: Optional[str] = Field(default=None)
    batch_number: Optional[str] = Field(default=None)
    lot_number: Optional[str] = Field(default=None)
    colata_number: Optional[str] = Field(default=None)
    cast_number: Optional[str] = Field(default=None)
    charge_number: Optional[str] = Field(default=None)
    coil_number: Optional[str] = Field(default=None)
    certificate_number: Optional[str] = Field(default=None)
    order_number: Optional[str] = Field(default=None)
    order_date: Optional[str] = Field(default=None)
    traceability_identifier_type: Optional[str] = Field(default=None)
    traceability_identifier_label: Optional[str] = Field(default=None)
    traceability_identifier_value: Optional[str] = Field(default=None)
    total_items_detected: int = Field(...)
    items: List[ExtractedItem] = Field(...)
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    raw_model_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    ai_analysis_remarks: Optional[str] = Field(default=None)
    stage_b_extraction_audit: Optional[str] = Field(default=None)
    is_compliant: Optional[bool] = Field(default=None)
    outcome: Optional[str] = Field(default=None)
    status: Optional[str] = Field(default=None)
    needs_review: bool = Field(default=False)
    review_reasons: List[str] = Field(default_factory=list)
    confidence_breakdown: Optional[Dict[str, float]] = Field(default=None)
    explanation: Optional[Dict[str, Any]] = Field(default=None)
    traceability_status: Optional[str] = Field(default=None)
    traceability_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    identifier_visibility_verified: Optional[bool] = Field(default=None)
    accepted_identifier_values: Dict[str, Any] = Field(default_factory=dict)
    raw_identifier_candidates: Dict[str, Any] = Field(default_factory=dict)
    analysis_id: Optional[int] = Field(default=None)
    auto_accept_evidence: Optional[Dict[str, Any]] = Field(default=None)
