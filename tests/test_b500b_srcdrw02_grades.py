from __future__ import annotations

import unittest

from app.domain.grade_registry import resolve_grade
from app.domain.spec_registry import get_spec, resolve_spec
from app.schemas.extraction import (
    ExtractedItem,
    MechanicalProperties,
    UniversalDocumentExtraction,
    ValidationResult,
)
from app.services.document_profiler import DocumentProfile
from app.services.review_policy import apply_review_policy
from app.services.traceability import TRACEABILITY_VERIFIED
from app.services.validator import validate_document


def _make_extraction(*, items: list[ExtractedItem], confidence_score: float = 0.87) -> UniversalDocumentExtraction:
    return UniversalDocumentExtraction(
        supplier_name="Demo Supplier",
        document_type="Inspection Certificate",
        certificate_date="2022-06-10",
        total_items_detected=len(items),
        items=items,
        confidence_score=confidence_score,
        raw_model_confidence=confidence_score,
    )


class B500BGradeRegistryTests(unittest.TestCase):
    def test_b500b_aliases_resolve_to_canonical(self):
        for raw in ("B 500 B", "B500B", "B 500B", "B500 B", "BST 500 B"):
            with self.subTest(raw=raw):
                resolution = resolve_grade(raw)
                self.assertEqual(resolution.status, "resolved")
                self.assertEqual(resolution.canonical, "B500B")
                self.assertEqual(resolution.family_group, "rebar")

    def test_b500b_spec_resolves(self):
        spec_resolution = resolve_spec(resolve_grade("B 500 B"))
        self.assertEqual(spec_resolution.status, "resolved")
        spec = get_spec("B500B")
        self.assertIsNotNone(spec)
        assert spec is not None
        self.assertEqual(spec.min_yield_mpa, 500.0)
        self.assertEqual(spec.min_tensile_mpa, 540.0)


class B500BValidationTests(unittest.TestCase):
    def test_observed_megasa_values_pass_validation(self):
        item = ExtractedItem(
            item_id="1",
            heat_number="4533354",
            grade="B 500 B",
            grade_provenance="labeled_field",
            mechanical_properties=MechanicalProperties(
                yield_strength_mpa=588.0,
                tensile_strength_mpa=691.0,
                elongation_percentage=None,
            ),
            row_confidence=0.9,
        )
        validated = validate_document(_make_extraction(items=[item]))
        row = validated.items[0]
        self.assertEqual(row.validation.outcome, "COMPLIANT")
        self.assertTrue(row.validation.is_compliant)
        self.assertNotEqual(row.validation.outcome, "EXPLICIT_UNMAPPED_GRADE")
        self.assertFalse(any(reason.startswith("unresolved_grade:") for reason in validated.review_reasons))

    def test_b500b_below_minimum_yield_is_non_compliant(self):
        item = ExtractedItem(
            item_id="1",
            heat_number="4533354",
            grade="B500B",
            mechanical_properties=MechanicalProperties(
                yield_strength_mpa=480.0,
                tensile_strength_mpa=600.0,
                elongation_percentage=8.0,
            ),
            row_confidence=0.9,
        )
        validated = validate_document(_make_extraction(items=[item]))
        self.assertEqual(validated.items[0].validation.outcome, "NON_COMPLIANT")


class SRCDRW02GradeRegistryTests(unittest.TestCase):
    def test_srcdrw02_aliases_resolve(self):
        for raw in ("SRCDRW02", "SRC DRW02", "SRC-DRW02"):
            with self.subTest(raw=raw):
                resolution = resolve_grade(raw)
                self.assertEqual(resolution.status, "resolved")
                self.assertEqual(resolution.canonical, "SRCDRW02")

    def test_srcdrw02_does_not_emit_unresolved_grade(self):
        item = ExtractedItem(
            item_id="10",
            heat_number="21103692",
            grade="SRCDRW02",
            grade_provenance="labeled_field",
            mechanical_properties=MechanicalProperties(
                yield_strength_mpa=310.0,
                tensile_strength_mpa=369.0,
                elongation_percentage=32.0,
            ),
            row_confidence=0.9,
        )
        validated = validate_document(_make_extraction(items=[item]))
        row = validated.items[0]
        self.assertEqual(row.validation.outcome, "COMPLIANT")
        self.assertNotEqual(row.validation.outcome, "EXPLICIT_UNMAPPED_GRADE")
        self.assertFalse(any(reason.startswith("unresolved_grade:") for reason in validated.review_reasons))
        self.assertFalse(any("explicit_unmapped_grade" in reason for reason in validated.review_reasons))

    def test_srcdrw02_observed_values_pass_sanity_validation(self):
        item = ExtractedItem(
            item_id="10",
            heat_number="21103692",
            grade="SRC DRW02",
            mechanical_properties=MechanicalProperties(
                yield_strength_mpa=305.0,
                tensile_strength_mpa=361.0,
                elongation_percentage=33.0,
            ),
            row_confidence=0.9,
        )
        validated = validate_document(_make_extraction(items=[item]))
        self.assertTrue(validated.items[0].validation.is_compliant)


class SoftAuditReviewPolicyTests(unittest.TestCase):
    def test_soft_visual_ambiguity_alone_does_not_force_review_when_compliant(self):
        item = ExtractedItem(
            item_id="1",
            heat_number="4533354",
            grade="B 500 B",
            weight_or_length="12.000 KG",
            mechanical_properties=MechanicalProperties(
                yield_strength_mpa=588.0,
                tensile_strength_mpa=691.0,
                elongation_percentage=8.4,
            ),
            validation=ValidationResult(is_compliant=True, deviations=[], outcome="COMPLIANT"),
            needs_review=False,
            row_confidence=0.9,
            traceability_status=TRACEABILITY_VERIFIED,
        )
        extraction = _make_extraction(items=[item], confidence_score=0.87)
        extraction.traceability_status = TRACEABILITY_VERIFIED
        extraction.review_reasons = ["visual ambiguity detected in row"]
        extraction.needs_review = True

        decision = apply_review_policy(
            extraction,
            profile=DocumentProfile(
                document_id="doc-demo",
                filename="demo.pdf",
                page_count=1,
                has_text_layer=False,
                text_density=0.0,
                blur_score=40.0,
                noise_score=30.0,
                table_presence_hint=True,
                quality_class="noisy_scan",
                reasons=["blur_high"],
            ),
            review_confidence_threshold=0.75,
        )

        self.assertFalse(decision.review_required)
        self.assertFalse(extraction.needs_review)
        self.assertEqual(extraction.status, "COMPLETED")
        self.assertIn("visual ambiguity detected in row", extraction.review_reasons)


if __name__ == "__main__":
    unittest.main()
