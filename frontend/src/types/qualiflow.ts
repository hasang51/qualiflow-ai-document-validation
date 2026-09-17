export interface MechanicalProperties {
  yield_strength_mpa: number | null
  tensile_strength_mpa: number | null
  elongation_percentage: number | null
}

export type ValidationOutcome =
  | 'COMPLIANT'
  | 'NON_COMPLIANT'
  | 'UNRESOLVED_SPEC'
  | 'UNSUPPORTED_SPEC_FAMILY'
  | 'EXPLICIT_UNMAPPED_GRADE'
  | 'MISSING_CRITICAL_FIELD_GRADE'
  | 'NOT_VALIDATED'
  | 'NEEDS_REVIEW'

export type TraceabilityStatus = 'VERIFIED' | 'UNVERIFIED'

export interface ValidationResult {
  is_compliant: boolean | null
  deviations: string[]
  outcome?: ValidationOutcome | null
  rule_evidence?: Record<string, unknown>[]
}

export interface GradeResolutionPayload {
  raw?: string
  normalized?: string
  status?: string
  canonical?: string | null
  candidates?: string[]
  family_group?: string | null
  dual_designation?: boolean
  reason?: string
  confidence?: number
  spec?: Record<string, unknown> | null
}

export interface ExtractedItem {
  item_id: string | null
  pipe_id?: string | null
  heat_number: string | null
  batch_number: string | null
  lot_number?: string | null
  colata_number?: string | null
  cast_number?: string | null
  charge_number?: string | null
  coil_number?: string | null
  certificate_number: string | null
  order_number: string | null
  order_date?: string | null
  traceability_identifier_type?: string | null
  traceability_identifier_label?: string | null
  traceability_identifier_value?: string | null
  product_name?: string | null
  product_details?: string | null
  grade: string | null
  weight_or_length: string | null
  dimensions?: string | null
  standards?: string[] | null
  classifications?: string[] | null
  chemical_composition?: Record<string, number | null> | null
  mechanical_properties: MechanicalProperties | null
  validation: ValidationResult | null
  row_confidence?: number | null
  needs_review?: boolean
  grade_resolution?: GradeResolutionPayload | null
  grade_provenance?: string | null
  traceability_status?: TraceabilityStatus | null
  traceability_confidence?: number | null
  identifier_visibility_verified?: boolean | null
  accepted_identifier_values?: Record<string, unknown>
  raw_identifier_candidates?: Record<string, unknown>
  source_page?: number | null
}

export interface ExtractionResponse {
  supplier_name: string
  document_type: string
  batch_number: string | null
  lot_number?: string | null
  colata_number?: string | null
  cast_number?: string | null
  charge_number?: string | null
  coil_number?: string | null
  certificate_number: string | null
  order_number: string | null
  order_date?: string | null
  traceability_identifier_type?: string | null
  traceability_identifier_label?: string | null
  traceability_identifier_value?: string | null
  certificate_date: string | null
  total_items_detected: number
  items: ExtractedItem[]
  confidence_score: number
  ai_analysis_remarks: string | null
  is_compliant: boolean | null
  outcome?: ValidationOutcome | null
  status?: 'PROCESSING' | 'COMPLETED' | 'FAILED' | 'NEEDS_REVIEW' | 'AUTO_ACCEPT' | null
  compliance_status?: string | null
  needs_review?: boolean
  review_reasons?: string[]
  confidence_breakdown?: Record<string, number> | null
  explanation?: Record<string, unknown> | null
  traceability_status?: TraceabilityStatus | null
  traceability_confidence?: number | null
  identifier_visibility_verified?: boolean | null
  accepted_identifier_values?: Record<string, unknown>
  raw_identifier_candidates?: Record<string, unknown>
  processing_decision?: string | null
  labeled_dates?: Array<{ label: string; value: string }>
  extraction_finalization?: { tokens?: string[]; traces?: Record<string, unknown>[] }
}

export interface HealthResponse {
  status: string
  service: string
  version: string
  model: string
  engine: string
  known_grades: string[]
  supported_documents: string[]
}

export type ComplianceState = 'compliant' | 'non-compliant' | 'not-validated' | 'needs-review'

export interface TokenResponse {
  access_token: string
  token_type: string
}

export interface MeResponse {
  id: number
  email: string
  created_at: string
}

export type JobStatus = 'queued' | 'processing' | 'succeeded' | 'failed' | 'cancelled'

export interface JobCreateResponse {
  job_id: string
  status: JobStatus
  poll_url: string
}

export interface JobStatusResponse {
  job_id: string
  status: JobStatus
  created_at: string
  updated_at: string
  started_at: string | null
  finished_at: string | null
  error_message: string | null
  trace_id: string | null
  analysis_id: number | null
}

export interface JobResultResponse {
  job_id: string
  status: JobStatus
  result: Record<string, unknown> | null
  error_message: string | null
  analysis_id: number | null
}

export type JobProgressPhase = JobStatus | 'uploading'

export interface AnalysisListItem {
  id: number
  document_id: number
  status: 'PROCESSING' | 'COMPLETED' | 'FAILED' | 'NEEDS_REVIEW'
  extraction_confidence: number | null
  global_is_compliant: boolean | null
  supplier_name: string | null
  document_type: string | null
  total_items_detected: number | null
  created_at: string
  updated_at: string
}

export interface AnalysisDetail {
  id: number
  document_id: number
  status: 'PROCESSING' | 'COMPLETED' | 'FAILED' | 'NEEDS_REVIEW'
  extraction_confidence: number | null
  global_is_compliant: boolean | null
  supplier_name: string | null
  document_type: string | null
  certificate_date: string | null
  total_items_detected: number | null
  ai_analysis_remarks: string | null
  raw_response_json: Record<string, unknown> | null
  preprocessing_meta_json: Record<string, unknown> | null
  error_message: string | null
  created_at: string
  updated_at: string
  extraction: ExtractionResponse | null
}
