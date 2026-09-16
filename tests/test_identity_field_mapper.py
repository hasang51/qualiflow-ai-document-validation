from __future__ import annotations

from app.services.identity_field_mapper import apply_identity_field_mapping
from app.services.row_shape_normalizer import collapse_alternative_classification_rows


class IdentityFieldMapperTests:
    def test_single_product_chemistry_mechanical_matrix(self):
        result = apply_identity_field_mapping(
            [
                {
                    "product_name": "WELDWIRE SG2 / BRONZEWIRE SG2",
                    "grade": "SG2",
                    "dimensions": "0.80 mm",
                    "standards": ["EN ISO 14341-A", "M21", "C1"],
                    "chemical_composition": {"C": 0.08},
                    "mechanical_properties": {"yield_strength_mpa": 470.0},
                }
            ],
            metadata={},
        )
        item = result.rows[0]
        assert item["product_name"] == "WELDWIRE SG2 / BRONZEWIRE SG2"
        assert item["grade"] == "SG2"
        assert item["dimensions"] == "0.80 mm"
        assert item["standards"] == ["EN ISO 14341-A", "M21", "C1"]
        assert item["chemical_composition"]["C"] == 0.08

    def test_repeated_product_rows_stay_independent(self):
        result = apply_identity_field_mapping(
            [
                {"item_id": "1", "heat_number": "H-100", "grade": "S355J2", "dimensions": "12 mm"},
                {"item_id": "2", "heat_number": "H-200", "grade": "S355J2", "dimensions": "16 mm"},
            ],
            metadata={"header_grade": "S355J2"},
        )
        assert len(result.rows) == 2
        assert result.rows[0]["heat_number"] == "H-100"
        assert result.rows[1]["heat_number"] == "H-200"
        assert result.rows[0]["dimensions"] == "12 mm"
        assert result.rows[1]["dimensions"] == "16 mm"
        assert result.rows[0]["grade"] == "S355J2"
        assert result.rows[0].get("product_name") is None

    def test_product_name_separate_from_grade_and_standard(self):
        result = apply_identity_field_mapping(
            [{"grade": "WELDWIRE SG2 / BRONZEWIRE SG2", "standards": ["EN ISO 14341-A"]}],
            metadata={},
        )
        item = result.rows[0]
        assert item["product_name"] == "WELDWIRE SG2 / BRONZEWIRE SG2"
        assert item["grade"] == "SG2"
        assert item["standards"] == ["EN ISO 14341-A"]

    def test_multiple_standards_preserved_and_not_used_as_product(self):
        result = apply_identity_field_mapping(
            [{"grade": "M21 / C1", "product_name": "EN ISO 14341-A"}],
            metadata={"product_description": "WELDWIRE SG2"},
        )
        item = result.rows[0]
        assert item["product_name"] == "WELDWIRE SG2"
        assert "M21" in (item["standards"] or [])
        assert "C1" in (item["standards"] or [])
        assert "EN ISO 14341-A" in (item["standards"] or [])
        assert item["grade"] == "SG2"
        assert item["product_name"] != "EN ISO 14341-A"
        assert item["grade"] not in {"M21", "C1", "EN ISO 14341-A"}

    def test_explicit_diameter_preserved_with_unit(self):
        result = apply_identity_field_mapping(
            [{"Diameter": "0.80 mm", "grade": "SG2"}],
            metadata={},
        )
        assert result.rows[0]["dimensions"] == "0.80 mm"
        assert result.rows[0]["grade"] == "SG2"

    def test_certificate_po_and_batch_stay_independent(self):
        result = apply_identity_field_mapping(
            [
                {
                    "certificate_number": "SYN-IC-1001",
                    "order_number": "PO-77001",
                    "batch_number": "SYN41001",
                    "grade": "SG2",
                }
            ],
            metadata={
                "certificate_number": "SYN-IC-1001",
                "order_number": "PO-77001",
                "batch_number": "SYN41001",
            },
        )
        item = result.rows[0]
        assert item["certificate_number"] == "SYN-IC-1001"
        assert item["order_number"] == "PO-77001"
        assert item["batch_number"] == "SYN41001"
        assert item["grade"] == "SG2"
        assert item.get("product_name") is None

    def test_ambiguous_product_identity_stays_null(self):
        result = apply_identity_field_mapping(
            [{"grade": "M21", "standards": ["C1"]}],
            metadata={},
        )
        item = result.rows[0]
        assert item["product_name"] is None
        assert item["grade"] is None
        assert "M21" in (item["standards"] or [])
        assert "C1" in (item["standards"] or [])

    def test_does_not_synthesize_grade_from_unmapped_standard(self):
        result = apply_identity_field_mapping(
            [{"grade": None, "standards": ["EN ISO 14341-A"]}],
            metadata={},
        )
        assert result.rows[0]["grade"] is None
        assert result.rows[0]["product_name"] is None
        assert result.rows[0]["standards"] == ["EN ISO 14341-A"]

    def test_domain_standard_alias_may_fill_grade(self):
        result = apply_identity_field_mapping(
            [{"grade": "EN 10255"}],
            metadata={},
        )
        assert result.rows[0]["grade"] == "EN 10255"
        assert result.rows[0]["product_name"] is None


def test_single_product_chemistry_mechanical_matrix():
    IdentityFieldMapperTests().test_single_product_chemistry_mechanical_matrix()


def test_repeated_product_rows_stay_independent():
    IdentityFieldMapperTests().test_repeated_product_rows_stay_independent()


def test_product_name_separate_from_grade_and_standard():
    IdentityFieldMapperTests().test_product_name_separate_from_grade_and_standard()


def test_multiple_standards_preserved_and_not_used_as_product():
    IdentityFieldMapperTests().test_multiple_standards_preserved_and_not_used_as_product()


def test_explicit_diameter_preserved_with_unit():
    IdentityFieldMapperTests().test_explicit_diameter_preserved_with_unit()


def test_certificate_po_and_batch_stay_independent():
    IdentityFieldMapperTests().test_certificate_po_and_batch_stay_independent()


def test_ambiguous_product_identity_stays_null():
    IdentityFieldMapperTests().test_ambiguous_product_identity_stays_null()


def test_does_not_synthesize_grade_from_unmapped_standard():
    IdentityFieldMapperTests().test_does_not_synthesize_grade_from_unmapped_standard()


def test_domain_standard_alias_may_fill_grade():
    IdentityFieldMapperTests().test_domain_standard_alias_may_fill_grade()


def test_classification_rows_collapse_without_guessing_product():
    rows = [
        {
            "item_id": "M21",
            "grade": "M21",
            "mechanical_properties": {
                "yield_strength_mpa": 470.0,
                "tensile_strength_mpa": 560.0,
                "elongation_percentage": 26.0,
            },
        },
        {
            "item_id": "C1",
            "grade": "C1",
            "mechanical_properties": {
                "yield_strength_mpa": 440.0,
                "tensile_strength_mpa": 530.0,
                "elongation_percentage": 26.0,
            },
        },
    ]
    collapsed = collapse_alternative_classification_rows(rows, metadata={})
    mapped = apply_identity_field_mapping(collapsed.rows, metadata={})
    assert len(mapped.rows) == 1
    assert mapped.rows[0]["product_name"] is None
    assert mapped.rows[0]["grade"] is None
    assert mapped.rows[0]["standards"] == ["M21", "C1"]
