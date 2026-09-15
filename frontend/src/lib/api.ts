import type {
  AnalysisDetail,
  AnalysisListItem,
  ExtractedItem,
  ExtractionResponse,
  HealthResponse,
  JobCreateResponse,
  JobProgressPhase,
  JobResultResponse,
  JobStatus,
  JobStatusResponse,
  MeResponse,
  MechanicalProperties,
  TokenResponse,
  TraceabilityStatus,
  ValidationResult,
  ValidationOutcome,
} from '../types/qualiflow'

export type { JobProgressPhase } from '../types/qualiflow'

const DEFAULT_BASE_URL = 'http://127.0.0.1:8000'
export const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL as string | undefined)?.trim() || DEFAULT_BASE_URL

export class ApiError extends Error {
  status: number

  constructor(message: string, status = 0) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

const TOKEN_KEY = 'qualiflow_access_token'

export function getStoredToken(): string | null {
  return localStorage.getItem(TOKEN_KEY)
}

export function setStoredToken(token: string) {
  localStorage.setItem(TOKEN_KEY, token)
}

export function clearStoredToken() {
  localStorage.removeItem(TOKEN_KEY)
}

function asObject(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return {}
  }
  return value as Record<string, unknown>
}

function asString(value: unknown): string | null {
  return typeof value === 'string' ? value : null
}

function asNumber(value: unknown): number | null {
  return typeof value === 'number' && !Number.isNaN(value) ? value : null
}

function asBoolean(value: unknown): boolean | null {
  return typeof value === 'boolean' ? value : null
}

function normalizeStatus(value: unknown): ExtractionResponse['status'] {
  const raw = typeof value === 'string' ? value.toUpperCase() : ''
  if (
    raw === 'PROCESSING' ||
    raw === 'COMPLETED' ||
    raw === 'FAILED' ||
    raw === 'NEEDS_REVIEW' ||
    raw === 'AUTO_ACCEPT'
  ) {
    return raw
  }
  return null
}

function normalizeTraceabilityStatus(value: unknown): TraceabilityStatus | null {
  const raw = typeof value === 'string' ? value.toUpperCase() : ''
  if (raw === 'VERIFIED' || raw === 'UNVERIFIED') return raw
  return null
}

function parseMechanicalProperties(value: unknown): MechanicalProperties | null {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return null
  }

  const source = asObject(value)
  return {
    yield_strength_mpa: asNumber(source.yield_strength_mpa),
    tensile_strength_mpa: asNumber(source.tensile_strength_mpa),
    elongation_percentage: asNumber(source.elongation_percentage),
  }
}

function normalizeOutcome(value: unknown): ValidationOutcome | null {
  const raw = typeof value === 'string' ? value.toUpperCase() : ''
  const legacyMap: Record<string, ValidationOutcome> = {
    RESOLVED_COMPLIANT: 'COMPLIANT',
    RESOLVED_NON_COMPLIANT: 'NON_COMPLIANT',
    UNKNOWN_GRADE: 'NEEDS_REVIEW',
    AMBIGUOUS_GRADE: 'NEEDS_REVIEW',
    EXTRACTION_UNCERTAIN: 'NEEDS_REVIEW',
    NOT_APPLICABLE: 'NOT_VALIDATED',
  }
  const normalized = legacyMap[raw] ?? raw
  if (
    normalized === 'COMPLIANT' ||
    normalized === 'NON_COMPLIANT' ||
    normalized === 'UNRESOLVED_SPEC' ||
    normalized === 'UNSUPPORTED_SPEC_FAMILY' ||
    normalized === 'EXPLICIT_UNMAPPED_GRADE' ||
    normalized === 'MISSING_CRITICAL_FIELD_GRADE' ||
    normalized === 'NOT_VALIDATED' ||
    normalized === 'NEEDS_REVIEW'
  ) {
    return normalized
  }
  return null
}

function parseValidation(value: unknown): ValidationResult | null {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return null
  }

  const source = asObject(value)
  const deviationsRaw = source.deviations
  const deviations = Array.isArray(deviationsRaw)
    ? deviationsRaw.filter((entry): entry is string => typeof entry === 'string')
    : []

  return {
    is_compliant: asBoolean(source.is_compliant),
    deviations,
    outcome: normalizeOutcome(source.outcome),
    rule_evidence: Array.isArray(source.rule_evidence)
      ? source.rule_evidence.filter((entry): entry is Record<string, unknown> => !!entry && typeof entry === 'object')
      : [],
  }
}

function parseLabeledDates(value: unknown): Array<{ label: string; value: string }> {
  if (!Array.isArray(value)) return []
  return value
    .map((entry) => {
      const source = asObject(entry)
      const label = asString(source.label) ?? ''
      const dateValue = asString(source.value) ?? ''
      return { label, value: dateValue }
    })
    .filter((entry) => entry.label || entry.value)
}

function parseExtractionFinalization(value: unknown): ExtractionResponse['extraction_finalization'] {
  const source = asObject(value)
  const tokensRaw = source.tokens
  const tracesRaw = source.traces
  const tokens = Array.isArray(tokensRaw) ? tokensRaw.filter((token): token is string => typeof token === 'string') : []
  const traces = Array.isArray(tracesRaw)
    ? tracesRaw.filter((trace): trace is Record<string, unknown> => !!trace && typeof trace === 'object')
    : []

  if (tokens.length === 0 && traces.length === 0) return undefined
  return { tokens, traces }
}

function parseItem(value: unknown): ExtractedItem {
  const source = asObject(value)
  return {
    item_id: asString(source.item_id),
    pipe_id: asString(source.pipe_id),
    heat_number: asString(source.heat_number),
    batch_number: asString(source.batch_number),
    lot_number: asString(source.lot_number),
    colata_number: asString(source.colata_number),
    cast_number: asString(source.cast_number),
    charge_number: asString(source.charge_number),
    coil_number: asString(source.coil_number),
    certificate_number: asString(source.certificate_number),
    order_number: asString(source.order_number),
    traceability_identifier_type: asString(source.traceability_identifier_type),
    traceability_identifier_label: asString(source.traceability_identifier_label),
    traceability_identifier_value: asString(source.traceability_identifier_value),
    grade: asString(source.grade),
    weight_or_length: asString(source.weight_or_length),
    mechanical_properties: parseMechanicalProperties(source.mechanical_properties),
    validation: parseValidation(source.validation),
    row_confidence: asNumber(source.row_confidence),
    needs_review: asBoolean(source.needs_review) ?? false,
    traceability_status: normalizeTraceabilityStatus(source.traceability_status),
    traceability_confidence: asNumber(source.traceability_confidence),
    identifier_visibility_verified: asBoolean(source.identifier_visibility_verified),
    accepted_identifier_values: asObject(source.accepted_identifier_values),
    raw_identifier_candidates: asObject(source.raw_identifier_candidates),
  }
}

export function extractAnalysisId(value: unknown): number | null {
  const source = asObject(value)
  const data = asObject(source.data)
  const candidates = [
    source.analysis_id,
    source.id,
    source.history_id,
    source.result_id,
    data.analysis_id,
    data.id,
    data.result_id,
  ]

  for (const candidate of candidates) {
    const num = asNumber(candidate)
    if (num !== null && Number.isInteger(num) && num > 0) {
      return num
    }
  }

  return null
}

export interface ExtractDocumentResult {
  extraction: ExtractionResponse
  analysisId: number | null
}

function parseExtractionResponse(value: unknown): ExtractionResponse {
  const source = asObject(value)
  const itemsRaw = Array.isArray(source.items) ? source.items : []
  const items = itemsRaw.map(parseItem)

  return {
    supplier_name: asString(source.supplier_name) ?? 'Unknown supplier',
    document_type: asString(source.document_type) ?? 'Unknown document',
    batch_number: asString(source.batch_number),
    lot_number: asString(source.lot_number),
    colata_number: asString(source.colata_number),
    cast_number: asString(source.cast_number),
    charge_number: asString(source.charge_number),
    coil_number: asString(source.coil_number),
    certificate_number: asString(source.certificate_number),
    order_number: asString(source.order_number),
    traceability_identifier_type: asString(source.traceability_identifier_type),
    traceability_identifier_label: asString(source.traceability_identifier_label),
    traceability_identifier_value: asString(source.traceability_identifier_value),
    certificate_date: asString(source.certificate_date),
    total_items_detected:
      asNumber(source.total_items_detected) ?? items.length,
    items,
    confidence_score: asNumber(source.confidence_score) ?? 0,
    ai_analysis_remarks: asString(source.ai_analysis_remarks),
    is_compliant: asBoolean(source.is_compliant),
    outcome: normalizeOutcome(source.outcome),
    status: normalizeStatus(source.status),
    compliance_status: asString(source.compliance_status),
    needs_review: asBoolean(source.needs_review) ?? false,
    review_reasons: Array.isArray(source.review_reasons)
      ? source.review_reasons.filter((x): x is string => typeof x === 'string')
      : [],
    confidence_breakdown:
      source.confidence_breakdown && typeof source.confidence_breakdown === 'object'
        ? (source.confidence_breakdown as Record<string, number>)
        : null,
    explanation:
      source.explanation && typeof source.explanation === 'object'
        ? (source.explanation as Record<string, unknown>)
        : null,
    traceability_status: normalizeTraceabilityStatus(source.traceability_status),
    traceability_confidence: asNumber(source.traceability_confidence),
    identifier_visibility_verified: asBoolean(source.identifier_visibility_verified),
    accepted_identifier_values: asObject(source.accepted_identifier_values),
    raw_identifier_candidates: asObject(source.raw_identifier_candidates),
    processing_decision: asString(source.processing_decision),
    labeled_dates: parseLabeledDates(source.labeled_dates),
    extraction_finalization: parseExtractionFinalization(source.extraction_finalization),
  }
}

function mapStatusToMessage(status: number, serverDetail?: string): string {
  const detail = serverDetail?.trim()
  if (detail) return detail

  if (status === 0) {
    return 'Cannot reach the backend API. Verify the server is running and VITE_API_BASE_URL is correct.'
  }
  if (status === 401) return 'Session expired. Please login again.'
  if (status === 409) return 'This email is already registered.'
  if (status === 415) return 'Only PDF files are accepted.'
  if (status === 422) return 'The PDF could not be processed. Verify the file and retry.'
  if (status === 425) return 'Job is still processing.'
  if (status === 502) return 'The extraction engine returned an upstream error. Please retry shortly.'
  if (status === 500) return 'The backend returned an internal server error.'
  return `Request failed with status ${status}.`
}

function formatErrorDetail(detail: unknown): string | undefined {
  if (typeof detail === 'string' && detail.trim()) {
    return detail.trim()
  }
  if (Array.isArray(detail)) {
    const messages = detail
      .map((entry) => {
        if (typeof entry === 'string') return entry
        if (entry && typeof entry === 'object') {
          const source = entry as { msg?: unknown; message?: unknown }
          if (typeof source.msg === 'string') return source.msg
          if (typeof source.message === 'string') return source.message
        }
        return null
      })
      .filter((value): value is string => Boolean(value))
    if (messages.length > 0) return messages.join('; ')
  }
  if (detail && typeof detail === 'object') {
    const source = detail as { message?: unknown; error?: unknown }
    if (typeof source.message === 'string' && source.message.trim()) return source.message.trim()
    if (typeof source.error === 'string' && source.error.trim()) return source.error.trim()
  }
  return undefined
}

function formatApiErrorPayload(payload: Record<string, unknown>): string | undefined {
  const fromDetail = formatErrorDetail(payload.detail)
  if (fromDetail) return fromDetail
  if (typeof payload.message === 'string' && payload.message.trim()) return payload.message.trim()
  if (typeof payload.error === 'string' && payload.error.trim()) return payload.error.trim()
  return undefined
}

async function readErrorMessage(response: Response): Promise<string | undefined> {
  let bodyText = ''
  try {
    bodyText = await response.text()
    if (!bodyText.trim()) return undefined

    try {
      const payload = JSON.parse(bodyText) as Record<string, unknown>
      return formatApiErrorPayload(payload) ?? bodyText.trim().slice(0, 500)
    } catch {
      return bodyText.trim().slice(0, 500)
    }
  } catch {
    return undefined
  }
}

async function requestJson<T>(path: string, init?: RequestInit, auth = false): Promise<T> {
  const headers = new Headers(init?.headers)
  headers.set('Accept', 'application/json')
  if (auth) {
    const token = getStoredToken()
    if (token) {
      headers.set('Authorization', `Bearer ${token}`)
    }
  }

  let response: Response
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      ...init,
      headers,
    })
  } catch {
    throw new ApiError(mapStatusToMessage(0), 0)
  }

  if (!response.ok) {
    const detail = await readErrorMessage(response)
    throw new ApiError(mapStatusToMessage(response.status, detail), response.status)
  }

  return (await response.json()) as T
}

export async function getHealth(): Promise<HealthResponse> {
  return requestJson<HealthResponse>('/health', { method: 'GET' })
}

const JOB_POLL_INTERVAL_MS = 2000
const JOB_POLL_TIMEOUT_MS = 10 * 60 * 1000

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => {
    window.setTimeout(resolve, ms)
  })
}

function authHeaders(): Headers {
  const headers = new Headers()
  const token = getStoredToken()
  if (token) {
    headers.set('Authorization', `Bearer ${token}`)
  }
  return headers
}

export async function uploadDocument(file: File, auth = true): Promise<JobCreateResponse> {
  const formData = new FormData()
  formData.append('file', file)

  const headers = auth ? authHeaders() : new Headers()

  let response: Response
  try {
    response = await fetch(`${API_BASE_URL}/api/v1/uploads`, {
      method: 'POST',
      body: formData,
      headers,
    })
  } catch {
    throw new ApiError(mapStatusToMessage(0), 0)
  }

  if (!response.ok && response.status !== 202) {
    const detail = await readErrorMessage(response)
    throw new ApiError(mapStatusToMessage(response.status, detail), response.status)
  }

  return (await response.json()) as JobCreateResponse
}

export async function getJobStatus(jobId: string): Promise<JobStatusResponse> {
  return requestJson<JobStatusResponse>(`/api/v1/jobs/${jobId}`, { method: 'GET' }, true)
}

export async function getJobResult(jobId: string): Promise<JobResultResponse> {
  return requestJson<JobResultResponse>(`/api/v1/jobs/${jobId}/result`, { method: 'GET' }, true)
}

function isTerminalJobStatus(status: JobStatus): boolean {
  return status === 'succeeded' || status === 'failed' || status === 'cancelled'
}

export async function extractDocumentAsync(
  file: File,
  auth = true,
  onProgress?: (phase: JobProgressPhase) => void,
): Promise<ExtractDocumentResult> {
  onProgress?.('uploading')
  const created = await uploadDocument(file, auth)
  const jobId = created.job_id

  const deadline = Date.now() + JOB_POLL_TIMEOUT_MS
  let status: JobStatus = created.status
  onProgress?.(status)

  while (!isTerminalJobStatus(status)) {
    if (Date.now() > deadline) {
      throw new ApiError('Extraction timed out. Please retry.', 504)
    }
    await sleep(JOB_POLL_INTERVAL_MS)
    const statusResponse = await getJobStatus(jobId)
    status = statusResponse.status
    onProgress?.(status)
  }

  const result = await getJobResult(jobId)

  if (status === 'failed') {
    throw new ApiError(result.error_message || 'Document processing failed.', 500)
  }
  if (status === 'cancelled') {
    throw new ApiError('Job was cancelled.', 409)
  }
  if (!result.result) {
    throw new ApiError('Job succeeded but returned no result.', 500)
  }

  onProgress?.('succeeded')
  return {
    extraction: parseExtractionResponse(result.result),
    analysisId: result.analysis_id ?? extractAnalysisId(result.result),
  }
}

/** @deprecated Use extractDocumentAsync for the async upload + worker pipeline. */
export async function extractDocument(file: File, auth = true): Promise<ExtractDocumentResult> {
  return extractDocumentAsync(file, auth)
}

export async function register(email: string, password: string): Promise<TokenResponse> {
  return requestJson<TokenResponse>(
    '/api/v1/auth/register',
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password }),
    },
    false,
  )
}

export async function login(email: string, password: string): Promise<TokenResponse> {
  return requestJson<TokenResponse>(
    '/api/v1/auth/login',
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password }),
    },
    false,
  )
}

export async function getMe(): Promise<MeResponse> {
  return requestJson<MeResponse>('/api/v1/auth/me', { method: 'GET' }, true)
}

export async function getAnalyses(): Promise<AnalysisListItem[]> {
  return requestJson<AnalysisListItem[]>('/api/v1/analyses', { method: 'GET' }, true)
}

function mergeExtractionWithRawResponse(
  extraction: unknown,
  rawResponseJson: Record<string, unknown> | null,
  preprocessingMeta: Record<string, unknown> | null = null,
): unknown {
  const base = asObject(extraction)
  const raw = asObject(rawResponseJson)
  const pre = asObject(preprocessingMeta)
  const preFinalization = asObject(pre.extraction_finalization)
  return {
    ...base,
    compliance_status: base.compliance_status ?? raw.compliance_status,
    status: base.status ?? raw.status,
    needs_review: base.needs_review ?? raw.needs_review,
    processing_decision: base.processing_decision ?? raw.processing_decision,
    labeled_dates: base.labeled_dates ?? raw.labeled_dates ?? pre.labeled_dates,
    extraction_finalization: base.extraction_finalization ?? raw.extraction_finalization ?? preFinalization,
  }
}

export async function getAnalysisById(id: number): Promise<AnalysisDetail> {
  const raw = await requestJson<AnalysisDetail>(`/api/v1/analyses/${id}`, { method: 'GET' }, true)
  return {
    ...raw,
    extraction: raw.extraction
      ? parseExtractionResponse(
          mergeExtractionWithRawResponse(raw.extraction, raw.raw_response_json, raw.preprocessing_meta_json),
        )
      : null,
  }
}

export async function downloadDocument(documentId: number): Promise<Blob> {
  const token = getStoredToken()
  const headers = new Headers()
  if (token) headers.set('Authorization', `Bearer ${token}`)

  const response = await fetch(`${API_BASE_URL}/api/v1/documents/${documentId}/download`, {
    method: 'GET',
    headers,
  })
  if (!response.ok) {
    const detail = await readErrorMessage(response)
    throw new ApiError(mapStatusToMessage(response.status, detail), response.status)
  }
  return response.blob()
}
