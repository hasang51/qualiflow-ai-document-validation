from __future__ import annotations

import unittest

from app.schemas.extraction import (
    ExtractedItem,
    MechanicalProperties,
    UniversalDocumentExtraction,
    ValidationResult,
)
from app.services.document_profiler import DocumentProfile
from app.services.extraction_finalizer import (
    _apply_confidence_exempt_decision,
    eligible_for_confidence_exempt_reconcile,
    finalize_decision,
    reconcile_final_document_decision,
)
from app.services.review_policy import (
    AUTO_ACCEPT_BLOCKING_EXACT,
    AUTO_ACCEPT_POLICY_VERSION,
    SEVERE_SCAN_QUALITY_TOKEN,
    apply_review_policy,
    collect_auto_accept_blockers_from_payload,
    evaluate_review_policy,
    is_auto_accept_blocking_reason,
    is_severe_scan_profile,
)
from app.services.traceability import TRACEABILITY_UNVERIFIED, TRACEABILITY_VERIFIED


def _severe_profile() -> DocumentProfile:
    return DocumentProfile(
        document_id="doc-severe",
        filename="severe.pdf",
        page_count=1,
        has_text_layer=False,
        text_density=0.0,
        blur_score=15.0,
        noise_score=50.0,
        table_presence_hint=True,
        quality_class="severe_scan",
        reasons=["blur_extreme", "noise_extreme"],
    )


def _clean_profile() -> DocumentProfile:
    return DocumentProfile(
        document_id="doc-clean",
        filename="clean.pdf",
        page_count=1,
        has_text_layer=True,
        text_density=0.4,
        blur_score=300.0,
        noise_score=5.0,
        table_presence_hint=False,
        quality_class="digital_clean",
        reasons=["text_layer_present"],
    )


def _compliant_item(
    *,
    grade: str = "S195",
    outcome: str = "COMPLIANT",
    is_compliant: bool | None = True,
    deviations: list[str] | None = None,
    needs_review: bool = False,
) -> ExtractedItem:
    return ExtractedItem(
        item_id="1",
        heat_number="H123",
        grade=grade,
        weight_or_length="12.000 KG",
        mechanical_properties=MechanicalProperties(
            yield_strength_mpa=258.0,
            tensile_strength_mpa=421.0,
            elongation_percentage=29.0,
        ),
        validation=ValidationResult(
            is_compliant=is_compliant,
            deviations=deviations or [],
            outcome=outcome,
        ),
        row_confidence=0.95,
        traceability_status=TRACEABILITY_VERIFIED,
        needs_review=needs_review,
    )


def _make_extraction(
    *,
    items: list[ExtractedItem],
    confidence_score: float = 0.95,
    needs_review: bool = False,
    review_reasons: list[str] | None = None,
) -> UniversalDocumentExtraction:
    return UniversalDocumentExtraction(
        supplier_name="Supplier",
        document_type="Mill Test Certificate",
        total_items_detected=len(items),
        items=items,
        confidence_score=confidence_score,
        raw_model_confidence=confidence_score,
        traceability_status=TRACEABILITY_VERIFIED,
        needs_review=needs_review,
        review_reasons=review_reasons or [],
    )


def _run_decision_pipeline(
    extraction: UniversalDocumentExtraction,
    *,
    profile: DocumentProfile | None = None,
) -> dict:
    decision = apply_review_policy(extraction, profile=profile or _clean_profile())
    finalize_decision(extraction)
    payload = extraction.model_dump(mode="python")
    if extraction.explanation:
        payload["explanation"] = dict(extraction.explanation)
    if decision.decision == "auto_accept" and decision.confidence_only_reasons:
        payload["review_reasons"] = list(decision.confidence_only_reasons)
        payload["needs_review"] = True
        _apply_confidence_exempt_decision(payload)
    result = reconcile_final_document_decision(payload)
    review_policy = decision.to_dict()
    if result.get("processing_decision") == "auto_accept":
        evidence = review_policy.get("auto_accept_evidence")
        if evidence:
            result["auto_accept_evidence"] = evidence
    return result


def _assert_not_silently_accepted(testcase: unittest.TestCase, result: dict) -> None:
    """Unsafe documents must not clear review without an explicit review decision."""

    testcase.assertNotEqual(result.get("processing_decision"), "auto_accept")
    testcase.assertTrue(
        result.get("needs_review") is True
        or result.get("review_required") is True
        or result.get("status") == "NEEDS_REVIEW",
        msg="document was silently cleared without review routing",
    )


class AutoAcceptBlockingContractTests(unittest.TestCase):
    """Contract tests for the shared safety gate (must exist before implementation)."""

    def test_severe_scan_token_is_blocking(self):
        self.assertTrue(is_auto_accept_blocking_reason(SEVERE_SCAN_QUALITY_TOKEN))

    def test_grade_and_alignment_tokens_are_blocking(self):
        for token in (
            "ambiguous_grade",
            "unresolved_grade",
            "explicit_unmapped_grade",
            "mechanical_table_alignment_uncertain",
            "traceability_identifier_ocr_uncertain",
            "traceability_identifier_conflict",
        ):
            with self.subTest(token=token):
                self.assertTrue(is_auto_accept_blocking_reason(token))

    def test_confidence_only_tokens_are_not_blocking(self):
        self.assertFalse(is_auto_accept_blocking_reason("confidence_below_threshold"))

    def test_collect_blockers_from_severe_scan_payload(self):
        payload = {
            "document_type": "Mill Test Certificate",
            "explanation": {
                "document_profile": {"quality_class": "severe_scan"},
            },
        }
        blockers = collect_auto_accept_blockers_from_payload(payload)
        self.assertIn(SEVERE_SCAN_QUALITY_TOKEN, blockers)

    def test_is_severe_scan_profile_detects_quality_bucket(self):
        self.assertTrue(is_severe_scan_profile({"quality_bucket": "severe_scan"}))
        self.assertFalse(is_severe_scan_profile({"quality_bucket": "digital_clean"}))

    def test_blocking_exact_set_includes_required_tokens(self):
        required = {
            SEVERE_SCAN_QUALITY_TOKEN,
            "ambiguous_grade",
            "mechanical_table_alignment_uncertain",
            "traceability_identifier_ocr_uncertain",
            "extraction_schema_invalid",
            "model_output_unusable",
        }
        self.assertTrue(required.issubset(AUTO_ACCEPT_BLOCKING_EXACT))


def _assert_auto_accept_evidence_contract(testcase: unittest.TestCase, evidence: dict) -> None:
    testcase.assertEqual(evidence.get("policy_version"), AUTO_ACCEPT_POLICY_VERSION)
    testcase.assertEqual(evidence.get("decision"), "auto_accept")
    testcase.assertIn("document_quality", evidence)
    testcase.assertIsInstance(evidence.get("critical_fields_present"), dict)
    testcase.assertTrue(all(evidence["critical_fields_present"].values()))
    testcase.assertIs(evidence.get("traceability_verified"), True)
    testcase.assertIs(evidence.get("row_alignment_verified"), True)
    testcase.assertIs(evidence.get("validation_passed"), True)
    testcase.assertIsInstance(evidence.get("confidence_summary"), dict)
    testcase.assertEqual(evidence.get("blocking_reasons"), [])
    testcase.assertIsInstance(evidence.get("confidence_only_reasons"), list)
    testcase.assertIsInstance(evidence.get("satisfied_criteria"), list)
    testcase.assertTrue(evidence["satisfied_criteria"])


class AutoAcceptEvidenceContractTests(unittest.TestCase):
    def test_review_decision_to_dict_includes_evidence_on_auto_accept(self):
        extraction = _make_extraction(
            items=[_compliant_item()],
            confidence_score=0.70,
            needs_review=True,
            review_reasons=["confidence_below_threshold"],
        )
        decision = apply_review_policy(extraction, profile=_clean_profile())
        review_dict = decision.to_dict()
        self.assertEqual(review_dict["decision"], "auto_accept")
        evidence = review_dict.get("auto_accept_evidence")
        self.assertIsInstance(evidence, dict)
        _assert_auto_accept_evidence_contract(self, evidence)

    def test_review_decision_to_dict_omits_evidence_when_not_auto_accept(self):
        extraction = _make_extraction(items=[_compliant_item()])
        decision = apply_review_policy(extraction, profile=_severe_profile())
        review_dict = decision.to_dict()
        self.assertNotEqual(review_dict["decision"], "auto_accept")
        self.assertNotIn("auto_accept_evidence", review_dict)

    def test_unverifiable_traceability_prevents_auto_accept_evidence(self):
        item = _compliant_item()
        item.traceability_status = TRACEABILITY_UNVERIFIED
        extraction = _make_extraction(
            items=[item],
            confidence_score=0.95,
            needs_review=False,
            review_reasons=[],
        )
        decision = apply_review_policy(extraction, profile=_clean_profile())
        review_dict = decision.to_dict()
        self.assertNotEqual(review_dict.get("decision"), "auto_accept")
        self.assertNotIn("auto_accept_evidence", review_dict)


class AutoAcceptSafetyRegressionTests(unittest.TestCase):
    def test_severe_scan_never_auto_accepts_via_apply_review_policy(self):
        extraction = _make_extraction(items=[_compliant_item()])
        decision = apply_review_policy(extraction, profile=_severe_profile())
        self.assertTrue(decision.review_required)
        self.assertNotEqual(decision.decision, "auto_accept")
        self.assertIn("quality_blocker:severe_scan", decision.structured_reasons)

    def test_severe_scan_never_auto_accepts_via_evaluate_review_policy(self):
        decision = evaluate_review_policy(
            extracted_json={
                "document_type": "Mill Test Certificate",
                "heat_number": "H1",
                "grade": "S355J2",
                "yield_strength_mpa": 380.0,
                "tensile_strength_mpa": 500.0,
                "elongation_percentage": 25.0,
            },
            confidence=0.95,
            validation_errors=[],
            document_profile={"quality_bucket": "severe_scan"},
        )
        self.assertEqual(decision["decision"], "review_required")
        self.assertIn("quality_blocker:severe_scan", decision["review_reasons"])

    def test_severe_scan_never_auto_accepts_after_reconcile(self):
        extraction = _make_extraction(
            items=[_compliant_item()],
            confidence_score=0.95,
            needs_review=True,
            review_reasons=["confidence_below_threshold"],
        )
        result = _run_decision_pipeline(extraction, profile=_severe_profile())
        _assert_not_silently_accepted(self, result)

    def test_explicit_unmapped_grade_blocks_auto_accept(self):
        item = _compliant_item(
            grade="MYSTERY-GRADE",
            is_compliant=None,
            outcome="EXPLICIT_UNMAPPED_GRADE",
            deviations=["Grade not mapped"],
            needs_review=True,
        )
        extraction = _make_extraction(
            items=[item],
            needs_review=True,
            review_reasons=["explicit_unmapped_grade"],
        )
        result = _run_decision_pipeline(extraction)
        _assert_not_silently_accepted(self, result)

    def test_ambiguous_grade_blocks_auto_accept(self):
        extraction = _make_extraction(
            items=[_compliant_item()],
            needs_review=True,
            review_reasons=["ambiguous_grade"],
        )
        result = _run_decision_pipeline(extraction)
        _assert_not_silently_accepted(self, result)

    def test_unresolved_grade_blocks_auto_accept(self):
        extraction = _make_extraction(
            items=[_compliant_item()],
            needs_review=True,
            review_reasons=["unresolved_grade"],
        )
        result = _run_decision_pipeline(extraction)
        _assert_not_silently_accepted(self, result)

    def test_unverified_traceability_blocks_auto_accept(self):
        extraction = _make_extraction(
            items=[_compliant_item()],
            confidence_score=0.95,
            needs_review=True,
            review_reasons=["traceability_unverified"],
        )
        result = _run_decision_pipeline(extraction)
        _assert_not_silently_accepted(self, result)
        self.assertFalse(eligible_for_confidence_exempt_reconcile(result))

    def test_ocr_uncertain_traceability_blocks_auto_accept(self):
        extraction = _make_extraction(
            items=[_compliant_item()],
            needs_review=True,
            review_reasons=["traceability_identifier_ocr_uncertain"],
        )
        result = _run_decision_pipeline(extraction)
        _assert_not_silently_accepted(self, result)

    def test_mechanical_alignment_uncertain_blocks_auto_accept(self):
        extraction = _make_extraction(
            items=[_compliant_item()],
            confidence_score=0.70,
            needs_review=True,
            review_reasons=[
                "mechanical_table_alignment_uncertain",
                "confidence_below_threshold",
            ],
        )
        result = _run_decision_pipeline(extraction)
        _assert_not_silently_accepted(self, result)
        reasons = list(result.get("review_reasons") or [])
        self.assertTrue(
            "mechanical_table_alignment_uncertain" in reasons
            or any("mechanical_table_alignment_uncertain" in str(reason) for reason in reasons)
        )

    def test_blocking_reason_survives_confidence_exempt_reconcile(self):
        extraction = _make_extraction(
            items=[_compliant_item()],
            confidence_score=0.70,
            needs_review=True,
            review_reasons=["traceability_unverified", "confidence_below_threshold"],
        )
        payload = extraction.model_dump(mode="python")
        self.assertFalse(eligible_for_confidence_exempt_reconcile(payload))
        result = reconcile_final_document_decision(payload)
        self.assertNotEqual(result.get("processing_decision"), "auto_accept")

    def test_auto_accept_includes_evidence_object(self):
        extraction = _make_extraction(
            items=[_compliant_item()],
            confidence_score=0.70,
            needs_review=True,
            review_reasons=["confidence_below_threshold"],
        )
        result = _run_decision_pipeline(extraction)
        self.assertEqual(result.get("processing_decision"), "auto_accept")
        evidence = result.get("auto_accept_evidence")
        self.assertIsInstance(evidence, dict)
        _assert_auto_accept_evidence_contract(self, evidence)
        self.assertIn("confidence_below_threshold", evidence["confidence_only_reasons"])

    def test_clean_document_still_auto_accepts_at_070(self):
        extraction = _make_extraction(
            items=[_compliant_item()],
            confidence_score=0.70,
            needs_review=True,
            review_reasons=["confidence_below_threshold"],
        )
        result = _run_decision_pipeline(extraction, profile=_clean_profile())
        self.assertEqual(result.get("processing_decision"), "auto_accept")
        self.assertIsInstance(result.get("auto_accept_evidence"), dict)

    def test_nested_hard_blocker_never_auto_accepts_after_finalizer(self):
        extraction = _make_extraction(
            items=[_compliant_item()],
            confidence_score=0.70,
            needs_review=True,
            review_reasons=["confidence_below_threshold"],
        )
        payload = extraction.model_dump(mode="python")
        payload["explanation"] = {
            "review_policy": {
                "structured_reasons": ["mechanical_table_alignment_uncertain"],
                "review_reasons": ["confidence_below_threshold"],
            }
        }
        self.assertFalse(eligible_for_confidence_exempt_reconcile(payload))
        _apply_confidence_exempt_decision(payload)
        result = reconcile_final_document_decision(payload)
        self.assertNotEqual(result.get("processing_decision"), "auto_accept")
        self.assertNotIn("confidence_below_threshold", result.get("review_reasons") or [])
        self.assertIn(
            "mechanical_table_alignment_uncertain",
            result["explanation"]["review_policy"]["structured_reasons"],
        )

    def test_severe_scan_profile_never_auto_accepts_after_finalizer(self):
        extraction = _make_extraction(
            items=[_compliant_item()],
            confidence_score=0.95,
            needs_review=True,
            review_reasons=["confidence_below_threshold"],
        )
        payload = extraction.model_dump(mode="python")
        payload["explanation"] = {"document_profile": {"quality_class": "severe_scan"}}
        self.assertFalse(eligible_for_confidence_exempt_reconcile(payload))
        result = reconcile_final_document_decision(payload)
        _assert_not_silently_accepted(self, result)

    def test_prefix_hard_blocker_survives_confidence_clean(self):
        extraction = _make_extraction(
            items=[_compliant_item()],
            confidence_score=0.70,
            needs_review=True,
            review_reasons=[
                "missing_critical_field:grade",
                "confidence_below_threshold",
            ],
        )
        payload = extraction.model_dump(mode="python")
        _apply_confidence_exempt_decision(payload)
        self.assertNotEqual(payload.get("processing_decision"), "auto_accept")
        self.assertIn("missing_critical_field:grade", payload["review_reasons"])
        self.assertNotIn("confidence_below_threshold", payload["review_reasons"])

    def test_schema_failure_tokens_never_auto_accept(self):
        extraction = _make_extraction(
            items=[_compliant_item()],
            confidence_score=1.0,
            needs_review=True,
            review_reasons=["extraction_schema_invalid", "model_output_unusable"],
        )
        payload = _run_decision_pipeline(extraction, profile=_clean_profile())
        self.assertNotEqual(payload.get("status"), "COMPLETED")
        self.assertTrue(payload.get("needs_review"))
        self.assertIsNone(payload.get("auto_accept_evidence"))
        self.assertTrue(is_auto_accept_blocking_reason("extraction_schema_invalid"))
        self.assertTrue(is_auto_accept_blocking_reason("model_output_unusable"))


if __name__ == "__main__":
    unittest.main()
