# QualiFlow Evaluation Report (Enhanced Layer)

- metadata: `data\gold\m2\metadata.csv`
- predictions: `data\gold\m2\predictions`

## Accuracy Metrics

| metric | value |
| --- | --- |
| raw_exact_accuracy | 1.0 |
| business_normalized_accuracy | 1.0 |
| accuracy_delta | 0.0 |
| harmless_normalization_accepts | 0 |
| true_mismatches | 0 |
| review_needed | 0 |
| critical_identifier_mismatches | 0 |

## Legacy Metrics (backward compatible)

| metric | value |
| --- | --- |
| n_documents | 6 |
| field_accuracy | 1.0 |
| critical_field_accuracy | 1.0 |
| document_type_accuracy | 1.0 |
| processing_decision_accuracy | 1.0 |
| review_rate | 1.0 |
| unsafe_auto_accept_rate | 0.0 |
| missing_required_field_rate | 0.0 |
| schema_failures | 1 |
| schema_failure_rate | 0.1667 |
| average_latency_ms | 13.5 |
| total_input_tokens | 55 |
| total_output_tokens | 25 |
| estimated_cost_usd | 0.0 |
| raw_exact_accuracy | 1.0 |
| business_normalized_accuracy | 1.0 |
| accuracy_delta | 0.0 |
| harmless_normalization_accepts | 0 |
| true_mismatches | 0 |
| review_needed | 0 |
| critical_identifier_mismatches | 0 |

## Matcher Breakdown

| matcher | count |
| --- | --- |
| critical_identifier_strict_match | 2 |
| grade_alias_match | 2 |
| normalized_exact_match | 1 |
| numeric_tolerance_match | 6 |
| semantic_equivalent_match | 12 |
| supplier_name_fuzzy_match | 6 |

## Examples: raw fail, business pass

- (none)

## Examples: both fail

- (none)

## Examples: critical identifier failures

- (none)

## Method

The evaluation layer reports both `raw_exact_accuracy` (literal string equality) and `business_normalized_accuracy` (field-aware deterministic matchers). Critical identifiers use strict matching only. This measures evaluation quality and extraction comparability — not model safety routing.

