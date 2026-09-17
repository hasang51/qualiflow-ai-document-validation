# QualiFlow Evaluation Report

- metadata: `data\gold\m2\metadata.csv`
- predictions: `data\gold\m2\predictions`
- documents evaluated: 6

## Metrics

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

## Method

String fields are compared by exact match after lowercase conversion, whitespace trimming, and internal whitespace collapse. Numeric fields use a tolerance when both gold and predicted values are numeric. Missing prediction fields are counted as incorrect whenever a gold value exists.

Processing decision accuracy is computed only when the gold annotation provides `processing_decision`, or a derivable `review_required` label. `unsafe_auto_accept_rate` measures gold review-required documents that the prediction marks as not requiring review.

## No Fabricated Metrics

This report only summarizes values computed from the provided metadata, ground truth JSON files, and prediction JSON files. Missing gold labels are skipped from metric denominators, and missing predictions are reported in `failure_cases.csv` instead of being filled with invented values.

Failure cases written: 0
