from __future__ import annotations

import unittest

from app.schemas.extraction import (
    ExtractedItem,
    MechanicalProperties,
    UniversalDocumentExtraction,
    ValidationResult,
)
from app.services.confidence import normalize_confidence
from app.services.document_profiler import DocumentProfile
from app.services.extraction_finalizer import (
    _apply_confidence_exempt_decision,
    eligible_for_confidence_exempt_reconcile,
    finalize_canonical_response,
    finalize_decision,
    finalize_decision_on_result,
    finalize_extraction_fields,
    qualifies_for_confidence_exempt_auto_accept,
    reconcile_final_document_decision,
)
from app.services.review_policy import apply_review_policy, evaluate_review_policy
from app.services.traceability import TRACEABILITY_VERIFIED, sanitize_result_for_api_boundary


def _verified_compliant_extraction(*, confidence_score: float = 0.70, item_id: str | None = "1") -> UniversalDocumentExtraction:
    item = ExtractedItem(
        item_id=item_id,
        heat_number="H123",
        grade="S195",
        weight_or_length="12.000 KG",
        mechanical_properties=MechanicalProperties(
            yield_strength_mpa=258.0,
            tensile_strength_mpa=421.0,
            elongation_percentage=29.0,
        ),
        validation=ValidationResult(is_compliant=True, deviations=[], outcome="COMPLIANT"),
        row_confidence=0.95,
        traceability_status=TRACEABILITY_VERIFIED,
    )
    return UniversalDocumentExtraction(
        supplier_name="Supplier",
        document_type="Mill Test Certificate",
        total_items_detected=1,
        items=[item],
        confidence_score=confidence_score,
        raw_model_confidence=confidence_score,
        traceability_status=TRACEABILITY_VERIFIED,
        needs_review=True,
        review_reasons=["confidence falls below threshold"],
    )


def _canonical_regression_payload() -> dict:
    return {
        "supplier_name": "Supplier",
        "document_type": "Mill Test Certificate",
        "traceability_status": TRACEABILITY_VERIFIED,
        "confidence_score": 0.70,
        "needs_review": True,
        "review_reasons": ["confidence_below_threshold"],
        "items": [
            {
                "item_id": None,
                "heat_number": "410537",
                "grade": "NOVOFIL SG2 / NOVOBRONZE SG2 - DIAM 0.80 mm P.L.W. BRONZE S-300 (Kg 1.080)",
                "weight_or_length": "1.080 Kg",
                "mechanical_properties": {
                    "yield_strength_mpa": 470.0,
                    "tensile_strength_mpa": 560.0,
                    "elongation_percentage": 26.0,
                },
                "validation": {
                    "is_compliant": True,
                    "deviations": [],
                    "outcome": "COMPLIANT",
                },
                "traceability_status": TRACEABILITY_VERIFIED,
                "needs_review": False,
            }
        ],
    }


class ExtractionFinalizerFieldTests(unittest.TestCase):
    def test_certificate_date_priority_over_po_date(self):
        metadata = {
            "certificate_date": "01/01/2020",
            "labeled_dates": [
                {"label": "PO Date", "value": "01/01/2020"},
                {"label": "Certificate Date", "value": "08/07/2024"},
            ],
        }
        result = finalize_extraction_fields(metadata, [])
        self.assertEqual(result.metadata["certificate_date"], "08/07/2024")
        self.assertIn("certificate_date:from_label:CERTIFICATE DATE", result.tokens)

    def test_certificate_date_normalizes_dot_format(self):
        metadata = {
            "labeled_dates": [{"label": "Date", "value": "27.04.2011"}],
        }
        result = finalize_extraction_fields(metadata, [])
        self.assertEqual(result.metadata["certificate_date"], "27/04/2011")

    def test_excluded_date_labels_are_ignored_when_only_option(self):
        metadata = {
            "labeled_dates": [
                {"label": "Delivery Date", "value": "15/03/2023"},
                {"label": "Analysis Timestamp", "value": "16/03/2023 10:00"},
            ],
        }
        result = finalize_extraction_fields(metadata, [])
        self.assertNotIn("certificate_date", result.metadata)
        self.assertNotIn("certificate_date:from_label", " ".join(result.tokens))

    def test_product_name_separated_from_short_grade(self):
        metadata = {"product_description": "WELDWIRE SG2 / BRONZEWIRE SG2"}
        rows = [{"grade": "SG2", "mechanical_properties": {}}]
        result = finalize_extraction_fields(metadata, rows)
        self.assertEqual(result.rows[0]["grade"], "SG2")
        self.assertEqual(result.rows[0]["product_name"], "WELDWIRE SG2 / BRONZEWIRE SG2")
        self.assertIn("product_name:from_metadata", result.tokens)

    def test_spec_line_not_used_as_grade_when_product_material_exists(self):
        metadata = {"product_description": "WELDWIRE SG2 / BRONZEWIRE SG2"}
        rows = [{"grade": "SG2", "EN ISO 9606": "EN ISO 9606-A", "mechanical_properties": {}}]
        result = finalize_extraction_fields(metadata, rows)
        self.assertEqual(result.rows[0]["grade"], "SG2")
        self.assertEqual(result.rows[0]["product_name"], "WELDWIRE SG2 / BRONZEWIRE SG2")
        self.assertNotEqual(result.rows[0]["grade"], result.rows[0].get("standards"))

    def test_spec_line_moves_to_standards_instead_of_overwriting_grade(self):
        metadata = {}
        rows = [{"grade": "304", "Description": "EN ISO 9606-A W 304", "mechanical_properties": {}}]
        result = finalize_extraction_fields(metadata, rows)
        self.assertEqual(result.rows[0]["grade"], "304")
        self.assertIsNone(result.rows[0].get("product_name"))
        self.assertIn("EN ISO 9606-A W 304", result.rows[0].get("standards") or [])

    def test_item_id_sequential_fallback(self):
        rows = [
            {"item_id": None, "grade": "S355"},
            {"item_id": "existing", "grade": "S355"},
            {"grade": "S355"},
        ]
        result = finalize_extraction_fields({}, rows)
        self.assertEqual(result.rows[0]["item_id"], "1")
        self.assertEqual(result.rows[1]["item_id"], "existing")
        self.assertEqual(result.rows[2]["item_id"], "3")
        self.assertIn("item_id_assigned:sequential", result.tokens)

    def test_item_id_placeholder_values_are_replaced(self):
        payload = {
            "items": [
                {"item_id": "—", "heat_number": "H1", "grade": "S355", "weight_or_length": "1 KG"},
            ]
        }
        finalize_canonical_response(payload)
        self.assertEqual(payload["items"][0]["item_id"], "1")

    def test_grade_detail_suffix_trimmed_on_canonical_response(self):
        payload = {
            "items": [
                {
                    "grade": "NOVOFIL SG2 / NOVOBRONZE SG2 - DIAM 0.80 mm P.L.W. BRONZE S-300 (Kg 1.080)",
                }
            ]
        }
        finalize_canonical_response(payload)
        self.assertEqual(payload["items"][0]["grade"], "SG2")
        self.assertEqual(payload["items"][0]["product_name"], "NOVOFIL SG2 / NOVOBRONZE SG2")
        self.assertEqual(payload["items"][0]["dimensions"], "0.80 mm")

    def test_canonical_missing_rate_zero_when_all_auto_accept_fields_present(self):
        payload = _canonical_regression_payload()
        result = finalize_canonical_response(payload)
        self.assertEqual(result.missing_critical_fields_rate, 0.0)


def _confidence_exempt_compliant_payload(
    *,
    review_reasons: list[str] | None = None,
    explanation: dict | None = None,
) -> dict:
    payload = _canonical_regression_payload()
    payload["review_required"] = True
    payload["status"] = "NEEDS_REVIEW"
    if review_reasons is not None:
        payload["review_reasons"] = review_reasons
    if explanation is not None:
        payload["explanation"] = explanation
    return payload


class ExtractionFinalizerDecisionTests(unittest.TestCase):
    def test_qualifies_for_confidence_exempt_auto_accept(self):
        extraction = _verified_compliant_extraction()
        self.assertTrue(qualifies_for_confidence_exempt_auto_accept(extraction))

    def test_missing_heat_number_not_eligible_without_traceability_identifier(self):
        extraction = _verified_compliant_extraction()
        extraction.items[0].heat_number = None
        extraction.items[0].batch_number = None
        extraction.traceability_identifier_value = None
        payload = extraction.model_dump(mode="python")
        self.assertFalse(qualifies_for_confidence_exempt_auto_accept(payload))

    def test_missing_weight_or_length_not_eligible(self):
        extraction = _verified_compliant_extraction()
        extraction.items[0].weight_or_length = None
        self.assertFalse(qualifies_for_confidence_exempt_auto_accept(extraction))

    def test_finalize_decision_clears_confidence_only_review(self):
        extraction = _verified_compliant_extraction(confidence_score=0.70)
        tokens = finalize_decision(extraction)
        self.assertIn("auto_accept:confidence_exempt", tokens)
        self.assertFalse(extraction.needs_review)
        self.assertNotIn("confidence falls below threshold", extraction.review_reasons)
        self.assertIsInstance(extraction.auto_accept_evidence, dict)
        self.assertIn("gates_passed", extraction.auto_accept_evidence)

    def test_auto_accept_at_070_after_review_policy(self):
        extraction = _verified_compliant_extraction(confidence_score=0.70)
        apply_review_policy(
            extraction,
            profile=DocumentProfile(
                document_id="doc",
                filename="x.pdf",
                page_count=1,
                has_text_layer=True,
                text_density=0.5,
                blur_score=300.0,
                noise_score=5.0,
                table_presence_hint=False,
                quality_class="digital_clean",
                reasons=[],
            ),
            review_confidence_threshold=0.75,
        )
        finalize_decision(extraction)
        self.assertFalse(extraction.needs_review)

    def test_confidence_normalization_does_not_block_when_eligible(self):
        extraction = _verified_compliant_extraction(confidence_score=0.70)
        assessment = normalize_confidence(
            extraction=extraction,
            preprocessing_meta={
                "extraction_finalization": {"missing_critical_fields_rate": 0.0},
            },
            raw_reported_total_items=1,
            review_confidence_threshold=0.75,
        )
        self.assertNotIn("confidence falls below threshold", assessment.review_reasons)
        self.assertEqual(assessment.metrics["confidence_breakdown"]["overall_decision_confidence"], 0.70)

    def test_evaluate_review_policy_confidence_exempt(self):
        decision = evaluate_review_policy(
            extracted_json={
                "document_type": "Mill Test Certificate",
                "traceability_status": TRACEABILITY_VERIFIED,
                "items": [
                    {
                        "heat_number": "H123",
                        "grade": "S195",
                        "weight_or_length": "12 KG",
                        "needs_review": False,
                        "mechanical_properties": {
                            "yield_strength_mpa": 258.0,
                            "tensile_strength_mpa": 421.0,
                            "elongation_percentage": 29.0,
                        },
                        "validation": {
                            "is_compliant": True,
                            "deviations": [],
                            "outcome": "COMPLIANT",
                        },
                    }
                ],
            },
            confidence={"overall_decision_confidence": 0.70, "_overall": 0.70},
            validation_errors=[],
            document_profile={"quality_class": "digital_clean"},
            confidence_threshold=0.75,
        )
        self.assertEqual(decision["decision"], "auto_accept")
        self.assertNotIn("confidence_below_threshold", decision["review_reasons"])

    def test_canonical_regression_missing_item_id_auto_accept_at_070(self):
        payload = _canonical_regression_payload()
        finalize_canonical_response(payload)
        finalize_decision_on_result(payload)

        self.assertEqual(payload["items"][0]["item_id"], "1")
        self.assertEqual(payload["items"][0]["grade"], "SG2")
        self.assertEqual(payload["items"][0]["product_name"], "NOVOFIL SG2 / NOVOBRONZE SG2")
        self.assertFalse(payload["needs_review"])
        self.assertFalse(payload["review_required"])
        self.assertEqual(payload["processing_decision"], "auto_accept")
        self.assertEqual(payload["compliance_status"], "COMPLIANT")
        self.assertNotIn("confidence_below_threshold", payload["review_reasons"])
        self.assertIsInstance(payload.get("auto_accept_evidence"), dict)

    def test_api_boundary_applies_canonical_finalization(self):
        extraction = _verified_compliant_extraction(confidence_score=0.70, item_id=None)
        extraction.review_reasons = ["confidence_below_threshold"]
        payload = sanitize_result_for_api_boundary(extraction)

        self.assertEqual(payload["items"][0]["item_id"], "1")
        self.assertFalse(payload["needs_review"])
        self.assertEqual(payload["processing_decision"], "auto_accept")
        self.assertEqual(payload["status"], "AUTO_ACCEPT")
        self.assertNotIn("confidence_below_threshold", payload["review_reasons"])
        self.assertIsInstance(payload.get("auto_accept_evidence"), dict)

    def test_reconcile_batch_traceability_auto_accept_with_heat_alias(self):
        payload = {
            "document_type": "Mill Test Certificate",
            "traceability_status": TRACEABILITY_VERIFIED,
            "confidence_score": 0.70,
            "needs_review": True,
            "status": "NEEDS_REVIEW",
            "review_reasons": ["confidence_below_threshold"],
            "explanation": {
                "review_policy": {
                    "decision": "review_required",
                    "review_required": True,
                    "needs_review": True,
                    "structured_reasons": ["confidence_below_threshold"],
                    "review_reasons": ["confidence_below_threshold"],
                    "evidence_gaps": ["confidence_below_threshold"],
                    "all_reasons": ["confidence_below_threshold", "confidence falls below threshold"],
                },
                "validation_outcome": {
                    "review_required": True,
                    "review_reasons": ["confidence_below_threshold"],
                },
            },
            "items": [
                {
                    "item_id": "1",
                    "heat_number": None,
                    "batch_number": "410537",
                    "traceability_identifier_value": "410537",
                    "grade": "NOVOFIL SG2 / NOVOBRONZE SG2",
                    "weight_or_length": "1.080 Kg",
                    "mechanical_properties": {
                        "yield_strength_mpa": 470.0,
                        "tensile_strength_mpa": 560.0,
                        "elongation_percentage": 26.0,
                    },
                    "validation": {
                        "is_compliant": True,
                        "deviations": [],
                        "outcome": "COMPLIANT",
                    },
                    "traceability_status": TRACEABILITY_VERIFIED,
                    "needs_review": False,
                }
            ],
        }
        reconcile_final_document_decision(payload)

        self.assertEqual(payload["heat_number"], "410537")
        self.assertEqual(payload["items"][0]["heat_number"], "410537")
        self.assertEqual(payload["items"][0]["batch_number"], "410537")
        self.assertFalse(payload["needs_review"])
        self.assertFalse(payload["review_required"])
        self.assertEqual(payload["status"], "AUTO_ACCEPT")
        self.assertEqual(payload["processing_decision"], "auto_accept")
        self.assertNotIn("confidence_below_threshold", payload["review_reasons"])
        self.assertIsInstance(payload.get("auto_accept_evidence"), dict)
        self.assertNotIn(
            "confidence_below_threshold",
            payload["explanation"]["review_policy"]["structured_reasons"],
        )
        self.assertNotIn(
            "confidence_below_threshold",
            payload["explanation"]["review_policy"]["all_reasons"],
        )
        self.assertEqual(payload["explanation"]["review_policy"]["decision"], "auto_accept")
        self.assertFalse(payload["explanation"]["review_policy"]["review_required"])

    def test_heat_alias_remarks_replace_misleading_no_heat_text(self):
        payload = {
            "ai_analysis_remarks": "No heat number is visible on this certificate.",
            "items": [
                {
                    "heat_number": None,
                    "batch_number": "410537",
                    "traceability_identifier_label": "COLATA/BATCH n°",
                    "traceability_identifier_value": "410537",
                }
            ],
        }
        reconcile_final_document_decision(payload)
        remarks = payload["ai_analysis_remarks"]
        self.assertNotIn("No heat number is visible", remarks)
        self.assertIn("No field explicitly labeled as heat number was visible", remarks)
        self.assertIn("Primary traceability identifier COLATA/BATCH n° 410537", remarks)
        self.assertIn("mapped to the canonical heat/batch field", remarks)

    def test_heat_alias_remarks_use_generic_field_label_when_no_traceability_label(self):
        payload = {
            "items": [
                {
                    "heat_number": None,
                    "lot_number": "LOT-24-01",
                }
            ],
        }
        reconcile_final_document_decision(payload)
        remarks = payload["ai_analysis_remarks"]
        self.assertIn("Primary traceability identifier LOT NO LOT-24-01", remarks)

    def test_nested_review_policy_decision_consistency_after_reconcile(self):
        payload = {
            "document_type": "Mill Test Certificate",
            "traceability_status": TRACEABILITY_VERIFIED,
            "confidence_score": 0.70,
            "needs_review": True,
            "review_required": True,
            "status": "NEEDS_REVIEW",
            "review_reasons": ["confidence_below_threshold"],
            "explanation": {
                "review_policy": {
                    "decision": "review_required",
                    "review_required": True,
                    "needs_review": True,
                    "structured_reasons": ["confidence_below_threshold"],
                    "review_reasons": ["confidence_below_threshold"],
                    "evidence_gaps": ["confidence_below_threshold"],
                    "all_reasons": ["confidence_below_threshold"],
                },
                "validation_outcome": {
                    "review_required": True,
                    "review_reasons": ["confidence_below_threshold"],
                },
            },
            "items": [
                {
                    "item_id": "1",
                    "heat_number": "410537",
                    "grade": "NOVOFIL SG2 / NOVOBRONZE SG2",
                    "weight_or_length": "1.080 Kg",
                    "mechanical_properties": {
                        "yield_strength_mpa": 470.0,
                        "tensile_strength_mpa": 560.0,
                        "elongation_percentage": 26.0,
                    },
                    "validation": {
                        "is_compliant": True,
                        "deviations": [],
                        "outcome": "COMPLIANT",
                    },
                    "traceability_status": TRACEABILITY_VERIFIED,
                    "needs_review": False,
                }
            ],
        }
        reconcile_final_document_decision(payload)

        self.assertFalse(payload["needs_review"])
        self.assertIn(payload["status"], {"COMPLETED", "AUTO_ACCEPT"})
        self.assertEqual(payload["explanation"]["review_policy"]["decision"], "auto_accept")
        self.assertFalse(payload["explanation"]["review_policy"]["review_required"])

    def test_hard_blocker_strips_confidence_but_never_auto_accepts(self):
        payload = _confidence_exempt_compliant_payload(
            review_reasons=["traceability_unverified", "confidence_below_threshold"],
        )
        _apply_confidence_exempt_decision(payload)
        self.assertNotEqual(payload.get("processing_decision"), "auto_accept")
        self.assertNotIn("confidence_below_threshold", payload["review_reasons"])
        self.assertIn("traceability_unverified", payload["review_reasons"])

    def test_nested_hard_blocker_blocks_confidence_exempt_reconcile(self):
        payload = _confidence_exempt_compliant_payload(
            review_reasons=["confidence_below_threshold"],
            explanation={
                "review_policy": {
                    "structured_reasons": ["traceability_unverified"],
                    "review_reasons": ["confidence_below_threshold"],
                }
            },
        )
        self.assertFalse(eligible_for_confidence_exempt_reconcile(payload))
        finalize_decision_on_result(payload)
        self.assertNotEqual(payload.get("processing_decision"), "auto_accept")
        self.assertNotIn("confidence_below_threshold", payload["review_reasons"])
        self.assertIn(
            "traceability_unverified",
            payload["explanation"]["review_policy"]["structured_reasons"],
        )

    def test_severe_scan_profile_blocks_confidence_exempt_reconcile(self):
        payload = _confidence_exempt_compliant_payload(
            review_reasons=["confidence_below_threshold"],
            explanation={"document_profile": {"quality_class": "severe_scan"}},
        )
        self.assertFalse(eligible_for_confidence_exempt_reconcile(payload))
        finalize_decision_on_result(payload)
        self.assertNotEqual(payload.get("processing_decision"), "auto_accept")


if __name__ == "__main__":
    unittest.main()
