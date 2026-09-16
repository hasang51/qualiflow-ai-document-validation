from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

FieldCategory = Literal["metadata", "row", "numeric", "text"]
ValueType = Literal["string", "number", "date", "enum", "boolean"]


@dataclass(frozen=True)
class CanonicalFieldDefinition:
    canonical_name: str
    description: str
    business_meaning: str
    category: FieldCategory
    value_type: ValueType
    units: str | None
    examples: tuple[str, ...]
    supported_header_synonyms: tuple[str, ...]


CANONICAL_FIELDS: dict[str, CanonicalFieldDefinition] = {
    "supplier_name": CanonicalFieldDefinition(
        canonical_name="supplier_name",
        description="Supplier/manufacturer name in certificate header.",
        business_meaning="The legal/commercial producer tied to the certificate.",
        category="metadata",
        value_type="string",
        units=None,
        examples=("TOSYALI Demir Celik", "ACME Steel"),
        supported_header_synonyms=("SUPPLIER", "MANUFACTURER", "PRODUCER", "COMPANY", "FIRMA"),
    ),
    "document_type": CanonicalFieldDefinition(
        canonical_name="document_type",
        description="High-level certificate/document type.",
        business_meaning="Document family/class used for downstream interpretation.",
        category="metadata",
        value_type="enum",
        units=None,
        examples=("Mill Test Certificate", "CoA"),
        supported_header_synonyms=("DOCUMENT TYPE", "CERTIFICATE TYPE", "DOC TYPE", "BELGE TIPI"),
    ),
    "certificate_date": CanonicalFieldDefinition(
        canonical_name="certificate_date",
        description="Issue/certificate date.",
        business_meaning="Document date for compliance timeline checks.",
        category="metadata",
        value_type="date",
        units=None,
        examples=("19.06.2020", "2024-01-31"),
        supported_header_synonyms=("DATE", "CERTIFICATE DATE", "ISSUE DATE", "TARIH"),
    ),
    "item_id": CanonicalFieldDefinition(
        canonical_name="item_id",
        description="Product/item profile identifier.",
        business_meaning="Identifies the line item profile or dimension family.",
        category="row",
        value_type="string",
        units=None,
        examples=("IPE-100X55", "KB-50X50X6000"),
        supported_header_synonyms=("ITEM", "SIZE", "EBAT", "PROFILE", "PROFIL"),
    ),
    "heat_number": CanonicalFieldDefinition(
        canonical_name="heat_number",
        description="Heat/melt reference number.",
        business_meaning="Traceability key linking product to melting batch.",
        category="row",
        value_type="string",
        units=None,
        examples=("812003694", "S1200594"),
        supported_header_synonyms=(
            "HEAT NO",
            "HEAT NUMBER",
            "HEAT#",
            "MELT NO",
            "MELT NUMBER",
            "SCHMELZE NO",
            "SCHMELZE NUMBER",
            "DOKUM NO",
            "DOKUM NUMARASI",
            "DÖKÜM NO",
            "DÖKÜM NUMARASI",
        ),
    ),
    "batch_number": CanonicalFieldDefinition(
        canonical_name="batch_number",
        description="Batch reference number.",
        business_meaning="Traceability key linking product to a production batch.",
        category="metadata",
        value_type="string",
        units=None,
        examples=("BATCH-1009", "B-2024-01"),
        supported_header_synonyms=(
            "BATCH NO",
            "BATCH NUMBER",
            "BATCH N",
            "BATCH N°",
        ),
    ),
    "lot_number": CanonicalFieldDefinition(
        canonical_name="lot_number",
        description="Lot reference number.",
        business_meaning="Traceability key linking product to lot identity.",
        category="metadata",
        value_type="string",
        units=None,
        examples=("LOT-001", "24-LOT-77"),
        supported_header_synonyms=("LOT", "LOT NO", "LOT NUMBER", "LOTTO", "N LOTTO"),
    ),
    "colata_number": CanonicalFieldDefinition(
        canonical_name="colata_number",
        description="Colata reference number.",
        business_meaning="Traceability key used on colata-based certificates.",
        category="metadata",
        value_type="string",
        units=None,
        examples=("410537", "COL-88"),
        supported_header_synonyms=("COLATA", "N COLATA", "COLATA/BATCH", "COLATA NO", "COLATA N°", "COLATA NUMBER"),
    ),
    "cast_number": CanonicalFieldDefinition(
        canonical_name="cast_number",
        description="Cast reference number.",
        business_meaning="Traceability key linking to cast identity.",
        category="metadata",
        value_type="string",
        units=None,
        examples=("CAST-1001", "C-778"),
        supported_header_synonyms=("CAST", "CAST NO", "CAST NUMBER"),
    ),
    "charge_number": CanonicalFieldDefinition(
        canonical_name="charge_number",
        description="Charge reference number.",
        business_meaning="Traceability key linking to charge identity.",
        category="metadata",
        value_type="string",
        units=None,
        examples=("CH-4221", "CHARGE-77"),
        supported_header_synonyms=("CHARGE", "CHARGE NO", "CHARGE NUMBER"),
    ),
    "coil_number": CanonicalFieldDefinition(
        canonical_name="coil_number",
        description="Coil reference number.",
        business_meaning="Traceability key for coil-level tracking.",
        category="row",
        value_type="string",
        units=None,
        examples=("COIL-17", "C001"),
        supported_header_synonyms=("COIL", "COIL NO", "COIL NUMBER"),
    ),
    "product_name": CanonicalFieldDefinition(
        canonical_name="product_name",
        description="Product or material identity as printed on the certificate.",
        business_meaning="Human-readable product/material name, distinct from grade and standards.",
        category="row",
        value_type="string",
        units=None,
        examples=("WELDWIRE SG2", "Structural beam IPE 100"),
        supported_header_synonyms=(
            "PRODUCT",
            "PRODUCT NAME",
            "PRODUCT DESCRIPTION",
            "MATERIAL NAME",
            "TRADE NAME",
            "DESCRIPTION",
        ),
    ),
    "grade": CanonicalFieldDefinition(
        canonical_name="grade",
        description="Material grade.",
        business_meaning="Steel/material grade used for compliance checks, not a standard label.",
        category="row",
        value_type="string",
        units=None,
        examples=("S275JR", "S355J2+N"),
        supported_header_synonyms=("GRADE", "MATERIAL", "KALITE", "STEEL GRADE", "MATERIAL GRADE"),
    ),
    "certificate_number": CanonicalFieldDefinition(
        canonical_name="certificate_number",
        description="Certificate identifier.",
        business_meaning="Document-level certificate number, independent of PO and heat/batch.",
        category="metadata",
        value_type="string",
        units=None,
        examples=("SYN-IC-1001", "CERT-2044"),
        supported_header_synonyms=(
            "CERTIFICATE NO",
            "CERTIFICATE NUMBER",
            "CERT NO",
            "CERT NUMBER",
            "CERTIFICATE N",
            "CERTIFICATO",
        ),
    ),
    "order_number": CanonicalFieldDefinition(
        canonical_name="order_number",
        description="Purchase order or customer order identifier.",
        business_meaning="Order/PO number, independent of certificate and heat/batch identifiers.",
        category="metadata",
        value_type="string",
        units=None,
        examples=("PO-77001", "ORD-88"),
        supported_header_synonyms=(
            "ORDER NO",
            "ORDER NUMBER",
            "PURCHASE ORDER",
            "PO NO",
            "PO NUMBER",
            "PO",
            "CUSTOMER ORDER",
        ),
    ),
    "weight_or_length": CanonicalFieldDefinition(
        canonical_name="weight_or_length",
        description="Weight/length-like quantity reported for line item.",
        business_meaning="Commercial quantity field that may be length or weight depending on document family.",
        category="row",
        value_type="string",
        units="mixed",
        examples=("12.000", "6.000", "12000 KG"),
        supported_header_synonyms=("WEIGHT", "LENGTH", "UZUNLUK", "AGIRLIK", "AĞIRLIK", "QTY"),
    ),
    "yield_strength_mpa": CanonicalFieldDefinition(
        canonical_name="yield_strength_mpa",
        description="Yield strength value.",
        business_meaning="Mechanical property minimum/actual yield strength.",
        category="numeric",
        value_type="number",
        units="MPa",
        examples=("314", "355"),
        supported_header_synonyms=("YIELD TS", "YIELD", "RE", "RP0.2", "RP 0.2", "RP 0,2"),
    ),
    "tensile_strength_mpa": CanonicalFieldDefinition(
        canonical_name="tensile_strength_mpa",
        description="Ultimate tensile strength value.",
        business_meaning="Mechanical property tensile strength used in validation.",
        category="numeric",
        value_type="number",
        units="MPa",
        examples=("463", "510"),
        supported_header_synonyms=("TENSILE TS", "TENSILE", "RM"),
    ),
    "elongation_percentage": CanonicalFieldDefinition(
        canonical_name="elongation_percentage",
        description="Elongation value as percent.",
        business_meaning="Ductility indicator used for material compliance.",
        category="numeric",
        value_type="number",
        units="%",
        examples=("29", "30"),
        supported_header_synonyms=("ELONGATION", "A%", "A %", "A"),
    ),
    "chemical_composition": CanonicalFieldDefinition(
        canonical_name="chemical_composition",
        description="Observed chemical composition values by element.",
        business_meaning="Element analysis used as structured evidence, not a compliance decision.",
        category="row",
        value_type="string",
        units=None,
        examples=("C=0.08", "Si=0.85"),
        supported_header_synonyms=("CHEMICAL COMPOSITION", "CHEMISTRY", "CHEMICAL ANALYSIS", "CHEM COMPOSITION"),
    ),
    "dimensions": CanonicalFieldDefinition(
        canonical_name="dimensions",
        description="Product diameter or dimensional size.",
        business_meaning="Physical size of the certified product when shown separately from mass/length.",
        category="row",
        value_type="string",
        units="mixed",
        examples=("0.80 mm", "12 mm"),
        supported_header_synonyms=(
            "DIAMETER",
            "DIAM",
            "DIA",
            "DIMENSION",
            "DIMENSIONS",
            "NOMINAL DIAMETER",
            "NOMINAL SIZE",
        ),
    ),
    "standards": CanonicalFieldDefinition(
        canonical_name="standards",
        description="Visible classification or standard labels.",
        business_meaning="Standards/classifications printed on the certificate for the product.",
        category="row",
        value_type="string",
        units=None,
        examples=("EN ISO 14341-A", "M21"),
        supported_header_synonyms=(
            "STANDARD",
            "STANDARDS",
            "CLASSIFICATION",
            "CLASSIFICATIONS",
            "NORMATIVE STANDARD",
            "NORM",
            "SPECIFICATION",
        ),
    ),
}


_TURKISH_CHAR_MAP = str.maketrans(
    {
        "Ç": "C",
        "Ğ": "G",
        "İ": "I",
        "I": "I",
        "Ö": "O",
        "Ş": "S",
        "Ü": "U",
        "ç": "c",
        "ğ": "g",
        "ı": "i",
        "i": "i",
        "ö": "o",
        "ş": "s",
        "ü": "u",
    }
)


def _normalize_for_matching(text: str) -> str:
    cleaned = text.translate(_TURKISH_CHAR_MAP)
    cleaned = cleaned.upper()
    cleaned = cleaned.strip()
    cleaned = cleaned.replace(",", ".")
    cleaned = re.sub(r"[/:;|]+", " ", cleaned)
    cleaned = re.sub(r"[_\-]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = re.sub(r"\bRP\s*0[\.,]?\s*2\b", "RP0.2", cleaned)
    cleaned = re.sub(r"\bHEAT\s*#\b", "HEAT NO", cleaned)
    cleaned = re.sub(r"\bA\s*%\b", "A%", cleaned)
    cleaned = re.sub(r"(?<=\w)\((.*?)\)", r" \1", cleaned)
    cleaned = re.sub(r"[^\w.%\s]+", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def normalize_header(text: str) -> str:
    """Normalize raw header text into deterministic matching form."""
    return _normalize_for_matching(text)


@lru_cache(maxsize=1)
def _synonym_index() -> dict[str, tuple[str, str]]:
    index: dict[str, tuple[str, str]] = {}
    for canonical_name, definition in CANONICAL_FIELDS.items():
        # Canonical key itself is always resolvable.
        canonical_norm = normalize_header(canonical_name)
        index[canonical_norm] = (canonical_name, canonical_name)
        for synonym in definition.supported_header_synonyms:
            index[normalize_header(synonym)] = (canonical_name, synonym)
    return index


def resolve_canonical_field(header_text: str) -> str | None:
    normalized = normalize_header(header_text)
    if not normalized:
        return None
    matched = _synonym_index().get(normalized)
    if matched is None:
        return None
    return matched[0]


def get_canonical_definition(field_name: str) -> dict[str, object] | None:
    definition = CANONICAL_FIELDS.get(field_name)
    if definition is None:
        return None
    normalized_variants = sorted(
        {normalize_header(variant) for variant in (definition.canonical_name, *definition.supported_header_synonyms)}
    )
    return {
        "canonical_name": definition.canonical_name,
        "description": definition.description,
        "business_meaning": definition.business_meaning,
        "category": definition.category,
        "value_type": definition.value_type,
        "units": definition.units,
        "examples": list(definition.examples),
        "supported_header_synonyms": list(definition.supported_header_synonyms),
        "normalized_header_variants": normalized_variants,
    }


def get_registry_snapshot() -> dict[str, dict[str, object]]:
    return {
        field_name: get_canonical_definition(field_name) or {}
        for field_name in sorted(CANONICAL_FIELDS)
    }


def explain_mapping(header_text: str) -> dict[str, object]:
    normalized = normalize_header(header_text)
    entry = _synonym_index().get(normalized)
    if entry is None:
        return {
            "input_header": header_text,
            "normalized_header": normalized,
            "matched_canonical_field": None,
            "matched_synonym": None,
            "confidence": 0.0,
            "reason": "unresolved header",
        }

    canonical_name, matched_synonym = entry
    reason = "exact synonym match"
    confidence = 1.0
    if matched_synonym == canonical_name and header_text.strip() != canonical_name:
        reason = "normalized canonical match"
        confidence = 0.95
    elif normalize_header(header_text) != normalize_header(matched_synonym):
        reason = "normalized synonym match"
        confidence = 0.9

    return {
        "input_header": header_text,
        "normalized_header": normalized,
        "matched_canonical_field": canonical_name,
        "matched_synonym": matched_synonym,
        "confidence": confidence,
        "reason": reason,
    }

