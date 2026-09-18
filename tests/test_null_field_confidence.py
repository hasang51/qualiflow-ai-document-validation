from __future__ import annotations

from app.services.extraction_pipeline import _mechanical_confidence, _normalize_row_dict


def test_null_mechanical_field_confidence_does_not_crash():
    row, _events = _normalize_row_dict(
        {
            "item_id": "1",
            "field_confidence": {
                "yield_strength_mpa": None,
                "tensile_strength_mpa": None,
                "elongation_percentage": None,
            },
            "mechanical_properties": {
                "yield_strength_mpa": 380.0,
                "tensile_strength_mpa": 500.0,
                "elongation_percentage": 25.0,
            },
        }
    )
    assert row["mechanical_properties"]["yield_strength_mpa"] is None
    assert row["mechanical_properties"]["tensile_strength_mpa"] is None
    assert row["mechanical_properties"]["elongation_percentage"] is None


def test_missing_field_confidence_keeps_mechanical_value():
    row, _events = _normalize_row_dict(
        {
            "item_id": "1",
            "mechanical_properties": {"yield_strength_mpa": 380.0},
        }
    )
    assert row["mechanical_properties"]["yield_strength_mpa"] == 380.0


def test_field_confidence_null_is_fail_closed():
    assert _mechanical_confidence({"yield_strength_mpa": None}, "yield_strength_mpa") == 0.0
    assert _mechanical_confidence({}, "yield_strength_mpa") == 1.0
    assert _mechanical_confidence({"yield_strength_mpa": 0.9}, "yield_strength_mpa") == 0.9
