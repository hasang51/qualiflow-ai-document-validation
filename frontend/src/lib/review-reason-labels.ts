const KNOWN_REVIEW_REASON_LABELS: Record<string, string> = {
  mechanical_table_alignment_uncertain: 'Mechanical table column alignment needs verification',
  traceability_unverified: 'Traceability identifiers require human verification',
  traceability_identifier_ocr_uncertain: 'Traceability identifier requires OCR verification',
  critical_identifier_unverified: 'Critical identifiers could not be verified with sufficient confidence',
  visual_ambiguity_detected_in_row: 'Visual ambiguity detected in the extracted row',
  'header_row_conflict:grade': 'Grade-related header conflict detected',
  'header_row_conflict:graderow0': 'Grade table header conflict detected',
  'row_shape:alternative_classification_rows_collapsed':
    'Multiple classification standards were merged into one product row',
  confidence_below_threshold: 'Overall confidence is below the automatic approval threshold',
  'confidence falls below threshold': 'Overall confidence is below the automatic approval threshold',
  unresolved_spec: 'Specification could not be resolved',
  unsupported_spec_family: 'Specification family is not supported',
  unsupported_document_type: 'Document type is not supported',
  validation_blocking_error: 'Validation blocking error',
  no_items_extracted: 'No line items extracted',
  table_found_but_no_rows: 'Table detected but no rows extracted',
  row_count_inconsistent: 'Reported vs extracted item count mismatch',
  explicit_unmapped_grade: 'Grade is explicitly stated but not mapped',
  unresolved_grade: 'Grade could not be resolved',
  ambiguous_grade: 'Grade is ambiguous',
  missing_critical_identifier_group: 'Missing critical traceability identifier group',
  heat_number_uncertain: 'Heat number is uncertain',
  weak_heat_number_evidence: 'Weak heat number evidence',
  low_identifier_confidence: 'Low identifier confidence',
}

const REVIEW_REASON_FIELD_LABELS: Record<string, string> = {
  yield_strength_mpa: 'yield strength',
  tensile_strength_mpa: 'tensile strength',
  elongation_percentage: 'elongation',
  heat_number: 'heat number',
  grade: 'grade',
}

const HEADER_ROW_CONFLICT_SUFFIX_LABELS: Record<string, string> = {
  grade: 'Grade-related header conflict detected',
  graderow0: 'Grade table header conflict detected',
}

function humanizeUnknownReason(code: string): string {
  return code
    .split(/[:_]/)
    .filter(Boolean)
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
    .join(' ')
}

export function formatReviewReason(code: string): string {
  const trimmed = code.trim()
  if (!trimmed) return trimmed

  const known =
    KNOWN_REVIEW_REASON_LABELS[trimmed] ?? KNOWN_REVIEW_REASON_LABELS[trimmed.toLowerCase()]
  if (known) return known

  const colonIndex = trimmed.indexOf(':')
  if (colonIndex > 0) {
    const prefix = trimmed.slice(0, colonIndex)
    const suffix = trimmed.slice(colonIndex + 1)
    const fieldLabel = REVIEW_REASON_FIELD_LABELS[suffix] ?? suffix.replace(/_/g, ' ')

    switch (prefix) {
      case 'missing_critical_field':
        return `Missing critical field: ${fieldLabel}`
      case 'conflicting_labeled_candidates':
        return `Conflicting labeled candidates: ${fieldLabel}`
      case 'low_confidence':
        return `Low confidence: ${fieldLabel}`
      case 'validation_conflict':
        return `Validation conflict: ${suffix.replace(/_/g, ' ')}`
      case 'numeric_uncertain':
        return `Uncertain numeric value: ${fieldLabel}`
      case 'header_row_conflict': {
        const headerLabel =
          HEADER_ROW_CONFLICT_SUFFIX_LABELS[suffix] ??
          HEADER_ROW_CONFLICT_SUFFIX_LABELS[suffix.toLowerCase()]
        return headerLabel ?? `Header row conflict: ${fieldLabel}`
      }
      case 'suspicious_duplication':
        return `Suspicious duplication: ${suffix.replace(/_/g, ' ')}`
      case 'document_quality':
        return `Document quality: ${suffix.replace(/_/g, ' ')}`
      case 'item_id_assigned':
        return `Item ID assigned: ${suffix.replace(/_/g, ' ')}`
      case 'unresolved_spec':
        return `Specification could not be resolved (${suffix.replace(/_/g, ' ')})`
      default:
        break
    }
  }

  if (/\s/.test(trimmed) && colonIndex <= 0) {
    return trimmed
  }

  if (/^[a-z][a-z0-9_]*$/i.test(trimmed)) {
    return humanizeUnknownReason(trimmed)
  }

  return trimmed
}

export function formatReviewReasons(codes: string[]): string[] {
  return codes.map(formatReviewReason).filter(Boolean)
}
