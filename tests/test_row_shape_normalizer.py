from __future__ import annotations

import unittest

from app.services.row_shape_normalizer import (
    backfill_single_item_context,
    collapse_alternative_classification_rows,
    collapse_vertical_mechanical_rows,
)


class RowShapeNormalizerTests(unittest.TestCase):
    def test_collapses_vertical_mechanical_rows_to_one_item(self):
        rows = [
            {
                "item_id": "M21 - Yield strength Re",
                "heat_number": None,
                "grade": "M21",
                "weight_or_length": None,
                "mechanical_properties": {
                    "yield_strength_mpa": 470.0,
                    "tensile_strength_mpa": None,
                    "elongation_percentage": None,
                },
                "row_confidence": 0.7,
            },
            {
                "item_id": "M21 - Tensile strength Rm",
                "heat_number": None,
                "grade": "M21",
                "weight_or_length": None,
                "mechanical_properties": {
                    "yield_strength_mpa": None,
                    "tensile_strength_mpa": 560.0,
                    "elongation_percentage": None,
                },
                "row_confidence": 0.7,
            },
            {
                "item_id": "M21 - Elongation A5d",
                "heat_number": None,
                "grade": "M21",
                "weight_or_length": None,
                "mechanical_properties": {
                    "yield_strength_mpa": None,
                    "tensile_strength_mpa": None,
                    "elongation_percentage": 26.0,
                },
                "row_confidence": 0.7,
            },
            {
                "item_id": "Product Details",
                "heat_number": None,
                "grade": None,
                "weight_or_length": "DIAM.0.80 mm (Kg.1.080)",
                "mechanical_properties": None,
                "row_confidence": 0.8,
            },
        ]
        result = collapse_vertical_mechanical_rows(
            rows,
            metadata={"heat_number": "410537", "header_grade": "NOVOFIL SG2/NOVOBRONZE SG2"},
        )

        self.assertEqual(len(result.rows), 1)
        self.assertEqual(result.rows[0]["heat_number"], "410537")
        self.assertEqual(result.rows[0]["grade"], "NOVOFIL SG2/NOVOBRONZE SG2")
        self.assertEqual(result.rows[0]["mechanical_properties"]["yield_strength_mpa"], 470.0)
        self.assertEqual(result.rows[0]["mechanical_properties"]["tensile_strength_mpa"], 560.0)
        self.assertEqual(result.rows[0]["mechanical_properties"]["elongation_percentage"], 26.0)
        self.assertIsNone(result.rows[0]["item_id"])
        self.assertIn("row_shape:vertical_mechanical_table_collapsed", result.tokens)

    def test_does_not_collapse_when_rows_have_heat_numbers(self):
        rows = [
            {
                "item_id": "Yield strength",
                "heat_number": "H1",
                "grade": "S355J2",
                "mechanical_properties": {"yield_strength_mpa": 355.0},
            },
            {
                "item_id": "Tensile strength",
                "heat_number": "H2",
                "grade": "S355J2",
                "mechanical_properties": {"tensile_strength_mpa": 500.0},
            },
        ]

        result = collapse_vertical_mechanical_rows(rows, metadata={})

        self.assertEqual(result.rows, rows)
        self.assertEqual(result.tokens, [])

    def test_backfills_single_item_metadata_context(self):
        result = backfill_single_item_context(
            [
                {
                    "item_id": "1",
                    "heat_number": None,
                    "grade": None,
                    "weight_or_length": None,
                    "mechanical_properties": {"yield_strength_mpa": 335.0},
                }
            ],
            metadata={"heat_number": "CH-22200", "header_grade": "1.4541/321"},
        )

        self.assertEqual(result.rows[0]["heat_number"], "CH-22200")
        self.assertEqual(result.rows[0]["grade"], "1.4541/321")
        self.assertIn("context_propagation:heat_number_from_metadata", result.tokens)

    def test_backfills_batch_and_certificate_for_single_item(self):
        result = backfill_single_item_context(
            [{"mechanical_properties": {"yield_strength_mpa": 470.0}}],
            metadata={
                "batch_number": "SYN41001",
                "certificate_number": "SYN-IC-1001",
                "order_number": "PO-77001",
                "field_confidence": {"batch_number": 0.96, "certificate_number": 0.95, "order_number": 0.94},
            },
        )

        self.assertEqual(result.rows[0]["batch_number"], "SYN41001")
        self.assertEqual(result.rows[0]["certificate_number"], "SYN-IC-1001")
        self.assertEqual(result.rows[0]["order_number"], "PO-77001")
        self.assertEqual(result.rows[0]["_identifier_confidence"]["batch_number"], 0.96)
        self.assertIn("context_propagation:batch_number_from_metadata", result.tokens)

    def test_collapses_alternative_classification_rows_for_one_product(self):
        rows = [
            {
                "item_id": "M21",
                "heat_number": None,
                "grade": "SG2",
                "weight_or_length": None,
                "mechanical_properties": {
                    "yield_strength_mpa": 470.0,
                    "tensile_strength_mpa": 560.0,
                    "elongation_percentage": 26.0,
                },
                "row_confidence": 0.9,
            },
            {
                "item_id": "C1",
                "heat_number": None,
                "grade": "SG2",
                "weight_or_length": None,
                "mechanical_properties": {
                    "yield_strength_mpa": 440.0,
                    "tensile_strength_mpa": 530.0,
                    "elongation_percentage": 26.0,
                },
                "row_confidence": 0.9,
            },
        ]

        result = collapse_alternative_classification_rows(
            rows,
            metadata={"heat_number": "410537"},
        )

        self.assertEqual(len(result.rows), 1)
        self.assertIsNone(result.rows[0]["item_id"])
        self.assertEqual(result.rows[0]["heat_number"], "410537")
        self.assertEqual(result.rows[0]["mechanical_properties"]["yield_strength_mpa"], 470.0)
        self.assertIn("row_shape:alternative_classification_rows_collapsed", result.tokens)

    def test_collapses_classification_labels_without_synthesizing_grade(self):
        rows = [
            {
                "item_id": None,
                "heat_number": "410537",
                "grade": "M21",
                "weight_or_length": None,
                "mechanical_properties": {
                    "yield_strength_mpa": 470.0,
                    "tensile_strength_mpa": 560.0,
                    "elongation_percentage": 26.0,
                },
                "row_confidence": 0.7,
            },
            {
                "item_id": None,
                "heat_number": "410537",
                "grade": "C1",
                "weight_or_length": None,
                "mechanical_properties": {
                    "yield_strength_mpa": 440.0,
                    "tensile_strength_mpa": 530.0,
                    "elongation_percentage": 26.0,
                },
                "row_confidence": 0.7,
            },
        ]

        result = collapse_alternative_classification_rows(
            rows,
            metadata={"heat_number": "410537"},
        )

        self.assertEqual(len(result.rows), 1)
        self.assertIsNone(result.rows[0]["grade"])
        self.assertEqual(result.rows[0]["heat_number"], "410537")
        self.assertEqual(result.rows[0]["standards"], ["M21", "C1"])

    def test_collapses_classifications_and_uses_explicit_product_grade(self):
        rows = [
            {
                "item_id": "M21",
                "heat_number": None,
                "grade": "M21",
                "mechanical_properties": {
                    "yield_strength_mpa": 470.0,
                    "tensile_strength_mpa": 560.0,
                    "elongation_percentage": 26.0,
                },
            },
            {
                "item_id": "C1",
                "heat_number": None,
                "grade": "C1",
                "mechanical_properties": {
                    "yield_strength_mpa": 440.0,
                    "tensile_strength_mpa": 530.0,
                    "elongation_percentage": 26.0,
                },
            },
        ]
        result = collapse_alternative_classification_rows(
            rows,
            metadata={"product_description": "WELDWIRE SG2", "heat_number": "410537"},
        )
        self.assertEqual(len(result.rows), 1)
        self.assertEqual(result.rows[0]["grade"], "SG2")
        self.assertEqual(result.rows[0]["standards"], ["M21", "C1"])


if __name__ == "__main__":
    unittest.main()
