from __future__ import annotations

import unittest

from app.domain.field_mapping_registry import (
    CANONICAL_FIELDS,
    explain_mapping,
    normalize_header,
    resolve_canonical_field,
)
from app.services.traceability import (
    TRACEABILITY_IDENTIFIER_GROUP_FIELDS,
    TRACEABILITY_IDENTIFIER_PRIORITY,
    validate_traceability,
)


def _is_traceability_field(field_name: str) -> bool:
    return field_name in TRACEABILITY_IDENTIFIER_GROUP_FIELDS


class NormalizeHeaderTests(unittest.TestCase):
    def test_lowercase_is_uppercased(self):
        self.assertEqual(normalize_header("heat no"), "HEAT NO")

    def test_turkish_characters_are_folded(self):
        self.assertEqual(normalize_header("DÖKÜM NO"), "DOKUM NO")

    def test_separators_are_collapsed(self):
        self.assertEqual(normalize_header(" heat/no :"), "HEAT NO")

    def test_rp02_variants_collapse_to_canonical_form(self):
        for text in ("RP 0.2", "RP 0,2", "RP0,2"):
            with self.subTest(text=text):
                self.assertEqual(normalize_header(text), "RP0.2")


class ResolveCanonicalFieldTests(unittest.TestCase):
    def test_canonical_name_is_resolvable(self):
        self.assertEqual(resolve_canonical_field("heat_number"), "heat_number")

    def test_english_synonym(self):
        self.assertEqual(resolve_canonical_field("HEAT NO"), "heat_number")

    def test_turkish_synonym_with_diacritics(self):
        self.assertEqual(resolve_canonical_field("DÖKÜM NO"), "heat_number")

    def test_german_synonym_schmelze_no(self):
        self.assertEqual(resolve_canonical_field("Schmelze No"), "heat_number")

    def test_yield_rp02_synonym(self):
        self.assertEqual(resolve_canonical_field("RP 0,2"), "yield_strength_mpa")

    def test_product_and_identifier_headers(self):
        self.assertEqual(resolve_canonical_field("Product Name"), "product_name")
        self.assertEqual(resolve_canonical_field("Certificate No"), "certificate_number")
        self.assertEqual(resolve_canonical_field("Purchase Order"), "order_number")
        self.assertEqual(resolve_canonical_field("Nominal Diameter"), "dimensions")
        self.assertEqual(resolve_canonical_field("Classification"), "standards")
        self.assertIsNone(resolve_canonical_field("NOT A FIELD"))

    def test_empty_string_is_none(self):
        self.assertIsNone(resolve_canonical_field(""))


class TraceabilityCanonicalMappingTests(unittest.TestCase):
    """Each traceability header variant maps to its specific canonical field."""

    TRACEABILITY_CASES = (
        ("Heat No", "heat_number"),
        ("Heat Number", "heat_number"),
        ("Heat#", "heat_number"),
        ("Batch No", "batch_number"),
        ("Batch Number", "batch_number"),
        ("Lot No", "lot_number"),
        ("Lot Number", "lot_number"),
        ("Cast No", "cast_number"),
        ("Cast Number", "cast_number"),
        ("Colata", "colata_number"),
        ("Colata No", "colata_number"),
        ("Colata Number", "colata_number"),
        ("Charge No", "charge_number"),
        ("Charge Number", "charge_number"),
    )

    def test_traceability_header_synonyms(self):
        for header, expected in self.TRACEABILITY_CASES:
            with self.subTest(header=header, expected=expected):
                self.assertEqual(resolve_canonical_field(header), expected)

    def test_traceability_fields_are_classified_as_traceability(self):
        for _, canonical in self.TRACEABILITY_CASES:
            with self.subTest(field=canonical):
                self.assertTrue(_is_traceability_field(canonical))

    def test_non_traceability_fields_are_not_classified_as_traceability(self):
        for field_name in (
            "grade",
            "supplier_name",
            "tensile_strength_mpa",
            "yield_strength_mpa",
            "elongation_percentage",
            "weight_or_length",
        ):
            with self.subTest(field=field_name):
                self.assertFalse(_is_traceability_field(field_name))

    def test_priority_order_preserves_specific_canonical_storage(self):
        """Display priority prefers heat, but verified lot_number stays in lot_number."""
        self.assertLess(
            TRACEABILITY_IDENTIFIER_PRIORITY.index("heat_number"),
            TRACEABILITY_IDENTIFIER_PRIORITY.index("lot_number"),
        )
        from app.schemas.extraction import ExtractedItem, MechanicalProperties, UniversalDocumentExtraction, ValidationResult

        item = ExtractedItem(
            item_id="ITEM-1",
            heat_number=None,
            lot_number="LOT-24-01",
            grade="S195",
            mechanical_properties=MechanicalProperties(
                yield_strength_mpa=264.0,
                tensile_strength_mpa=418.0,
                elongation_percentage=34.0,
            ),
            validation=ValidationResult(is_compliant=True, deviations=[], outcome="COMPLIANT"),
            row_confidence=0.95,
        )
        extraction = UniversalDocumentExtraction(
            supplier_name="Supplier",
            document_type="Mill Test Certificate",
            product_category="PIPE",
            lot_number="LOT-24-01",
            total_items_detected=1,
            items=[item],
            confidence_score=0.95,
            raw_model_confidence=0.95,
            is_compliant=True,
            outcome="COMPLIANT",
            status="COMPLETED",
        )
        meta = {
            "identifier_confidence": {
                "metadata": {"lot_number": 0.95},
                "rows": [{"lot_number": 0.94, "item_id": 0.93}],
            }
        }
        result = validate_traceability(extraction, meta)
        row = result.items[0]
        self.assertEqual(row.lot_number, "LOT-24-01")
        self.assertEqual(row.traceability_identifier_type, "lot_number")
        self.assertEqual(row.traceability_identifier_value, "LOT-24-01")
        self.assertIsNone(row.heat_number)


class ExplainMappingTests(unittest.TestCase):
    def test_explanation_contains_canonical_target(self):
        result = explain_mapping("HEAT NO")
        self.assertIsInstance(result, dict)
        self.assertEqual(result.get("matched_canonical_field"), "heat_number")
        self.assertEqual(result.get("matched_synonym"), "HEAT NO")

    def test_explanation_for_unresolved_header(self):
        result = explain_mapping("NOT A REAL HEADER")
        self.assertIsNone(result.get("matched_canonical_field"))
        self.assertEqual(result.get("confidence"), 0.0)


class CanonicalFieldsIntegrityTests(unittest.TestCase):
    def test_every_canonical_field_resolves_to_itself(self):
        for canonical_name in CANONICAL_FIELDS:
            with self.subTest(field=canonical_name):
                self.assertEqual(resolve_canonical_field(canonical_name), canonical_name)

    def test_every_declared_synonym_resolves_to_its_canonical_name(self):
        for canonical_name, definition in CANONICAL_FIELDS.items():
            for synonym in definition.supported_header_synonyms:
                with self.subTest(field=canonical_name, synonym=synonym):
                    self.assertEqual(resolve_canonical_field(synonym), canonical_name)


if __name__ == "__main__":
    unittest.main()
