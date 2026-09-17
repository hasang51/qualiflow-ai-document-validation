from __future__ import annotations

import unittest

from app.services.mechanical_table_mapper import (
    apply_mechanical_table_mapping,
    canonicalize_element_symbol,
    map_chemical_table_rows,
    map_mechanical_table_rows,
)


def test_canonicalize_element_symbol():
    assert canonicalize_element_symbol("C") == "C"
    assert canonicalize_element_symbol("carbon") == "C"
    assert canonicalize_element_symbol("Si") == "Si"
    assert canonicalize_element_symbol("silicon") == "Si"
    assert canonicalize_element_symbol("unknown") is None


class MechanicalTableMapperTests(unittest.TestCase):
    def test_maps_mtc_table_rows_using_results_column(self):
        table_rows = [
            {
                "property": "Proof Strength Rp0.2",
                "Specified": ">=205",
                "Results": 335,
            },
            {
                "property": "Proof Strength Rp1.0",
                "Results": 381,
            },
            {
                "property": "Tensile Strength Rm",
                "Specified": "515-750",
                "Min": 515,
                "Max": 750,
                "Results": 638,
            },
            {
                "property": "Elongation",
                "Results": "52/51",
            },
        ]

        result = map_mechanical_table_rows(table_rows)

        self.assertEqual(result.mechanical_properties["yield_strength_mpa"], 335.0)
        self.assertEqual(result.mechanical_properties["tensile_strength_mpa"], 638.0)
        self.assertEqual(result.mechanical_properties["elongation_percentage"], 52.0)
        self.assertEqual(result.auxiliary.get("yield_strength_rp1_0_mpa"), 381.0)
        self.assertEqual(result.auxiliary.get("elongation_raw"), "52/51")
        self.assertFalse(result.uncertain)

    def test_does_not_use_min_or_specified_for_tensile(self):
        table_rows = [
            {
                "property": "Tensile Strength Rm",
                "Specified": "515-750",
                "Min": 581,
                "Max": 750,
                "Results": 638,
            },
        ]

        result = map_mechanical_table_rows(table_rows)

        self.assertEqual(result.mechanical_properties["tensile_strength_mpa"], 638.0)

    def test_prefers_rp0_2_over_rp1_0_for_yield(self):
        table_rows = [
            {"property": "Proof Strength Rp1.0", "Results": 381},
            {"property": "Proof Strength Rp0.2", "Results": 335},
        ]

        result = map_mechanical_table_rows(table_rows)

        self.assertEqual(result.mechanical_properties["yield_strength_mpa"], 335.0)
        self.assertEqual(result.auxiliary.get("yield_strength_rp1_0_mpa"), 381.0)

    def test_does_not_map_proof_strength_to_tensile(self):
        table_rows = [
            {"property": "Proof Strength Rp0.2", "Results": 335},
        ]

        result = map_mechanical_table_rows(table_rows)

        self.assertIsNone(result.mechanical_properties["tensile_strength_mpa"])
        self.assertEqual(result.mechanical_properties["yield_strength_mpa"], 335.0)

    def test_marks_uncertain_when_conflicting_result_columns(self):
        table_rows = [
            {
                "property": "Tensile Strength Rm",
                "Result": 638,
                "Results": 581,
            },
        ]

        result = map_mechanical_table_rows(table_rows)

        self.assertTrue(result.uncertain)
        self.assertIn("mechanical_table_alignment_uncertain", result.tokens)

    def test_synthesizes_item_when_only_mechanical_table_rows_present(self):
        item_payload = {
            "mechanical_table_rows": [
                {"property": "Proof Strength Rp0.2", "Results": 470},
                {"property": "Tensile Strength Rm", "Results": 560},
                {"property": "Elongation", "Results": 26},
            ]
        }

        updated_rows, tokens, trace = apply_mechanical_table_mapping(
            [],
            item_payload=item_payload,
            items_raw=[],
        )

        self.assertEqual(len(updated_rows), 1)
        mp = updated_rows[0]["mechanical_properties"]
        self.assertEqual(mp["yield_strength_mpa"], 470.0)
        self.assertEqual(mp["tensile_strength_mpa"], 560.0)
        self.assertEqual(mp["elongation_percentage"], 26.0)
        self.assertTrue(trace.get("synthesized_item_from_mechanical_table"))

    def test_apply_mechanical_table_mapping_overwrites_item_row(self):
        rows = [
            {
                "heat_number": "H-100",
                "mechanical_properties": {
                    "yield_strength_mpa": 300.0,
                    "tensile_strength_mpa": 581.0,
                    "elongation_percentage": 50.0,
                },
            }
        ]
        item_payload = {
            "mechanical_table_rows": [
                {"property": "Proof Strength Rp0.2", "Results": 335},
                {"property": "Proof Strength Rp1.0", "Results": 381},
                {"property": "Tensile Strength Rm", "Min": 581, "Results": 638},
                {"property": "Elongation", "Results": "52/51"},
            ]
        }

        updated_rows, tokens, trace = apply_mechanical_table_mapping(
            rows,
            item_payload=item_payload,
            items_raw=[],
        )

        mp = updated_rows[0]["mechanical_properties"]
        self.assertEqual(mp["yield_strength_mpa"], 335.0)
        self.assertEqual(mp["tensile_strength_mpa"], 638.0)
        self.assertEqual(mp["elongation_percentage"], 52.0)
        self.assertEqual(trace["strategy"], "mechanical_table_row_column_alignment")
        self.assertEqual(tokens, [])


class ChemicalTableMapperTests(unittest.TestCase):
    def test_maps_chemistry_results_and_ignores_specified(self):
        table_rows = [
            {"property": "C", "Specified": "0.06-0.14", "Results": 0.08},
            {"property": "Si", "Min": 0.50, "Max": 1.00, "Results": 0.85},
            {"property": "Mn", "Results": 1.45},
        ]

        result = map_chemical_table_rows(table_rows)

        self.assertEqual(result.chemical_composition["C"], 0.08)
        self.assertEqual(result.chemical_composition["Si"], 0.85)
        self.assertEqual(result.chemical_composition["Mn"], 1.45)
        self.assertFalse(result.uncertain)

    def test_does_not_use_specified_when_results_missing(self):
        table_rows = [{"property": "C", "Specified": "0.06-0.14", "Min": 0.06}]

        result = map_chemical_table_rows(table_rows)

        self.assertNotIn("C", result.chemical_composition)

    def test_maps_wide_element_row(self):
        result = map_chemical_table_rows([{"C": 0.08, "Si": 0.85, "Mn": 1.45, "P": 0.015}])
        self.assertEqual(result.chemical_composition["C"], 0.08)
        self.assertEqual(result.chemical_composition["Si"], 0.85)

    def test_synthesizes_item_from_chemistry_when_items_empty(self):
        item_payload = {
            "chemical_table_rows": [
                {"property": "C", "Results": 0.08},
                {"property": "Si", "Results": 0.85},
            ]
        }

        updated_rows, tokens, trace = apply_mechanical_table_mapping(
            [],
            item_payload=item_payload,
            items_raw=[],
        )

        self.assertEqual(len(updated_rows), 1)
        self.assertEqual(updated_rows[0]["chemical_composition"]["C"], 0.08)
        self.assertTrue(trace.get("synthesized_item_from_chemical_table"))
        self.assertEqual(tokens, [])

    def test_synthesizes_item_from_chemistry_and_mechanicals(self):
        item_payload = {
            "mechanical_table_rows": [
                {"property": "Proof Strength Rp0.2", "Results": 470},
                {"property": "Tensile Strength Rm", "Results": 560},
                {"property": "Elongation", "Results": 26},
            ],
            "chemical_table_rows": [{"property": "C", "Results": 0.08, "source_page": 1}],
        }

        updated_rows, tokens, trace = apply_mechanical_table_mapping(
            [],
            item_payload=item_payload,
            items_raw=[],
        )

        self.assertEqual(len(updated_rows), 1)
        self.assertEqual(updated_rows[0]["mechanical_properties"]["yield_strength_mpa"], 470.0)
        self.assertEqual(updated_rows[0]["chemical_composition"]["C"], 0.08)
        self.assertEqual(updated_rows[0]["source_page"], 1)
        self.assertTrue(trace.get("synthesized_item_from_mechanical_table"))
        self.assertTrue(trace.get("synthesized_item_from_chemical_table"))


if __name__ == "__main__":
    unittest.main()
