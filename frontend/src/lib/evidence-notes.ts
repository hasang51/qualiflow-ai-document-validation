import { formatDecisionLabel, formatNullable, resolveProcessingDecisionValue } from './format'
import {
  dedupeExactStrings,
  dedupeReasons,
  formatDisplayReviewReason,
  formatValidationOutcome,
} from './presentation-safety'
import type { ExtractionResponse, ExtractedItem } from '../types/qualiflow'

export { formatReviewReason } from './review-reason-labels'

export interface EvidenceNoteSection {
  title: string
  lines: string[]
}

const PLACEHOLDER_ITEM_IDS = new Set(['', '-', '—', 'n/a', 'na', 'none', 'null'])

function isSequentialItemId(itemId: string, index: number): boolean {
  const normalized = itemId.trim()
  if (/^\d+$/.test(normalized)) {
    return Number(normalized) === index + 1
  }

  const match = normalized.toLowerCase().match(/^(?:item|row)[-_ ]?(\d+)$/)
  return match !== null && Number(match[1]) === index + 1
}

function itemRef(item: ExtractedItem, index: number): string {
  const itemId = item.item_id?.trim()
  if (!itemId || PLACEHOLDER_ITEM_IDS.has(itemId.toLowerCase()) || isSequentialItemId(itemId, index)) {
    return `Row ${index + 1}`
  }

  return itemId
}

function resolveCertificateDateSourceLabel(data: ExtractionResponse): string | null {
  const tokens = data.extraction_finalization?.tokens ?? []
  for (const token of tokens) {
    if (token.startsWith('certificate_date:from_label:')) {
      return token.slice('certificate_date:from_label:'.length)
    }
    if (token === 'certificate_date:from_metadata') {
      return 'document metadata'
    }
  }

  for (const trace of data.extraction_finalization?.traces ?? []) {
    if (trace.step !== 'certificate_date') continue
    const selectedLabel = typeof trace.selected_label === 'string' ? trace.selected_label : null
    if (selectedLabel) return selectedLabel
    if (trace.source === 'certificate_date') return 'document metadata'
  }

  if (!data.certificate_date) return null

  const normalizedDate = data.certificate_date.trim().toLowerCase()
  for (const entry of data.labeled_dates ?? []) {
    if (entry.value.trim().toLowerCase() === normalizedDate && entry.label) {
      return entry.label
    }
  }

  return null
}

function buildCertificateDateNotes(data: ExtractionResponse): string[] {
  if (!data.certificate_date) {
    return ['Not extracted']
  }

  const sourceLabel = resolveCertificateDateSourceLabel(data)
  if (sourceLabel) {
    return [`${data.certificate_date} (from ${sourceLabel})`]
  }

  return [data.certificate_date]
}

function buildTraceabilityNotes(data: ExtractionResponse): string[] {
  const lines: string[] = []

  if (data.traceability_identifier_label || data.traceability_identifier_value) {
    lines.push(
      `Document: ${formatNullable(data.traceability_identifier_label, 'Identifier')} ${formatNullable(data.traceability_identifier_value)}`,
    )
  }

  data.items.forEach((item, index) => {
    if (!item.traceability_identifier_label && !item.traceability_identifier_value) return
    lines.push(
      `${itemRef(item, index)}: ${formatNullable(item.traceability_identifier_label, 'Identifier')} ${formatNullable(item.traceability_identifier_value)}`,
    )
  })

  if (lines.length === 0) {
    return ['Not extracted']
  }

  return lines
}

function buildProductNotes(data: ExtractionResponse): string[] {
  const lines = data.items
    .map((item, index) => (item.product_name ? `${itemRef(item, index)}: ${item.product_name}` : null))
    .filter((line): line is string => line !== null)

  if (lines.length === 0) {
    return ['Not extracted']
  }

  return lines
}

function buildGradeNotes(data: ExtractionResponse): string[] {
  const lines = data.items
    .map((item, index) => (item.grade ? `${itemRef(item, index)}: ${item.grade}` : null))
    .filter((line): line is string => line !== null)

  if (lines.length === 0) {
    return ['Not extracted']
  }

  return lines
}

function buildStandardsNotes(data: ExtractionResponse): string[] {
  const lines = data.items
    .map((item, index) =>
      item.standards && item.standards.length > 0 ? `${itemRef(item, index)}: ${item.standards.join(', ')}` : null,
    )
    .filter((line): line is string => line !== null)

  if (lines.length === 0) {
    return ['Not extracted']
  }

  return lines
}

function buildDimensionNotes(data: ExtractionResponse): string[] {
  const lines = data.items
    .map((item, index) => (item.dimensions ? `${itemRef(item, index)}: ${item.dimensions}` : null))
    .filter((line): line is string => line !== null)

  if (lines.length === 0) {
    return ['Not extracted']
  }

  return lines
}

function buildValidationNotes(data: ExtractionResponse): string[] {
  const lines: string[] = []

  data.items.forEach((item, index) => {
    const validation = item.validation
    if (!validation) return

    const outcome = validation.outcome ?? (validation.is_compliant === true
      ? 'COMPLIANT'
      : validation.is_compliant === false
        ? 'NON_COMPLIANT'
        : 'NOT_VALIDATED')
    const ref = itemRef(item, index)
    const deviations = validation.deviations ?? []

    const outcomeLabel = formatValidationOutcome(outcome)
    const formattedDeviations = dedupeReasons(deviations).map(formatDisplayReviewReason)

    if (formattedDeviations.length === 0) {
      lines.push(`${ref}: ${outcomeLabel}`)
      return
    }

    lines.push(`${ref}: ${outcomeLabel} (${formattedDeviations.join('; ')})`)
  })

  if (data.outcome && lines.length === 0) {
    lines.push(`Document: ${formatValidationOutcome(data.outcome)}`)
  }

  if (lines.length === 0) {
    return ['None recorded']
  }

  return lines
}

function buildDecisionNotes(data: ExtractionResponse): string[] {
  const decision = resolveProcessingDecisionValue(data)
  const lines = [formatDecisionLabel(decision)]

  if (data.review_reasons && data.review_reasons.length > 0) {
    lines.push(...dedupeReasons(data.review_reasons).map(formatDisplayReviewReason))
  }

  return dedupeExactStrings(lines)
}

export function buildEvidenceNoteSections(data: ExtractionResponse): EvidenceNoteSection[] {
  return [
    { title: 'Certificate Date', lines: dedupeExactStrings(buildCertificateDateNotes(data)) },
    { title: 'Traceability Identifier', lines: dedupeExactStrings(buildTraceabilityNotes(data)) },
    { title: 'Product', lines: dedupeExactStrings(buildProductNotes(data)) },
    { title: 'Grade', lines: dedupeExactStrings(buildGradeNotes(data)) },
    { title: 'Standards', lines: dedupeExactStrings(buildStandardsNotes(data)) },
    { title: 'Dimensions', lines: dedupeExactStrings(buildDimensionNotes(data)) },
    { title: 'Validation', lines: dedupeExactStrings(buildValidationNotes(data)) },
    { title: 'Decision', lines: dedupeExactStrings(buildDecisionNotes(data)) },
  ]
}

export function hasModelRemarks(data: ExtractionResponse): boolean {
  return Boolean(data.ai_analysis_remarks?.trim())
}
