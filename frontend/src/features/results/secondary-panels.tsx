import { CheckCheck, Clipboard, Download, FileJson } from 'lucide-react'
import { useMemo, useState } from 'react'
import { Button } from '../../components/ui/button'
import { Card } from '../../components/ui/card'
import { buildEvidenceNoteSections } from '../../lib/evidence-notes'
import {
  dedupeReasons,
  formatDisplayReviewReason,
  getNoDeviationsMessage,
  hasNoReliableLineItems,
} from '../../lib/presentation-safety'
import type { ExtractionResponse, ExtractedItem } from '../../types/qualiflow'

const ITEM_IDENTIFIER_LINE_PATTERN =
  /secondary identifier|identifier candidate|candidate identifier|\bitem\s*id\b|\bpipe\s*coil\s*id\b/i

const MODEL_REMARKS_FALLBACK =
  'Model remarks unavailable because no reliable line items were extracted.'

function hasExplicitPipeOrItemIdentifiers(items: ExtractedItem[]): boolean {
  return items.some((item) => {
    const record = item as ExtractedItem & { pipe_coil_id?: string | null }
    return [item.item_id, item.pipe_id, record.pipe_coil_id].some(
      (value) => typeof value === 'string' && value.trim() !== '',
    )
  })
}

function sanitizeModelRemarks(text: string): string | null {
  const lines = text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)

  const filtered = lines.filter((line) => !ITEM_IDENTIFIER_LINE_PATTERN.test(line))
  const sanitized = filtered.join('\n').trim()
  return sanitized || null
}

interface SecondaryPanelsProps {
  data: ExtractionResponse
}

function downloadTextFile(filename: string, content: string, mimeType: string) {
  const blob = new Blob([content], { type: mimeType })
  const url = URL.createObjectURL(blob)
  const anchor = document.createElement('a')
  anchor.href = url
  anchor.download = filename
  anchor.click()
  URL.revokeObjectURL(url)
}

export function SecondaryPanels({ data }: SecondaryPanelsProps) {
  const [copied, setCopied] = useState(false)
  const rawJson = useMemo(() => JSON.stringify(data, null, 2), [data])
  const evidenceSections = useMemo(() => buildEvidenceNoteSections(data), [data])
  const sanitizedModelRemarks = useMemo(() => {
    const raw = data.ai_analysis_remarks?.trim()
    if (!raw) return null

    if (hasExplicitPipeOrItemIdentifiers(data.items)) {
      return raw
    }

    return sanitizeModelRemarks(raw)
  }, [data.ai_analysis_remarks, data.items])

  const allDeviations = useMemo(() => {
    const seen = new Set<string>()
    const result: { itemId: string; message: string }[] = []
    data.items.forEach((item, index) => {
      const itemId = item.item_id || `row-${index + 1}`
      for (const deviation of dedupeReasons(item.validation?.deviations ?? [])) {
        const message = formatDisplayReviewReason(deviation)
        const key = `${itemId}::${message}`
        if (seen.has(key)) continue
        seen.add(key)
        result.push({ itemId, message })
      }
    })
    return result
  }, [data.items])

  const noDeviationsMessage = useMemo(() => getNoDeviationsMessage(data), [data])

  async function copyJson() {
    await navigator.clipboard.writeText(rawJson)
    setCopied(true)
    setTimeout(() => setCopied(false), 1200)
  }

  function downloadJson() {
    downloadTextFile('qualiflow-extraction.json', rawJson, 'application/json')
  }

  return (
    <section className="grid gap-4 xl:grid-cols-3">
      <Card className="space-y-3 xl:col-span-1">
        <p className="text-xs uppercase tracking-wider text-slate-400">Evidence-Based Notes</p>
        <ul className="space-y-3 text-sm text-slate-200">
          {evidenceSections.map((section) => (
            <li key={section.title}>
              <p className="text-xs font-medium uppercase tracking-wide text-slate-500">{section.title}</p>
              <ul className="mt-1 list-disc space-y-0.5 pl-4 leading-relaxed text-slate-300">
                {section.lines.map((line, index) => (
                  <li key={`${section.title}-${index}`}>{line}</li>
                ))}
              </ul>
            </li>
          ))}
        </ul>

        {(sanitizedModelRemarks || hasNoReliableLineItems(data)) && (
          <details className="rounded-lg border border-slate-800 bg-slate-950/70">
            <summary className="cursor-pointer list-none px-3 py-2 text-xs uppercase tracking-wider text-slate-500">
              Model Remarks
            </summary>
            <p className="whitespace-pre-wrap border-t border-slate-800 px-3 py-2 text-sm leading-relaxed text-slate-400">
              {sanitizedModelRemarks ?? MODEL_REMARKS_FALLBACK}
            </p>
          </details>
        )}
      </Card>

      <Card className="space-y-3 xl:col-span-1">
        <p className="text-xs uppercase tracking-wider text-slate-400">Validation Findings</p>
        {allDeviations.length > 0 ? (
          <ul className="max-h-60 space-y-2 overflow-auto text-sm text-slate-200">
            {allDeviations.map((deviation, index) => (
              <li key={`${deviation.itemId}-${index}`} className="rounded-md border border-slate-800 bg-slate-900 p-2">
                <span className="mr-2 text-slate-500">[{deviation.itemId}]</span>
                {deviation.message}
              </li>
            ))}
          </ul>
        ) : (
          <p className="text-sm text-slate-300">{noDeviationsMessage}</p>
        )}
      </Card>

      <Card className="space-y-3 xl:col-span-1">
        <p className="text-xs uppercase tracking-wider text-slate-400">Raw JSON</p>
        <div className="flex flex-wrap gap-2">
          <Button variant="secondary" onClick={copyJson} className="h-9">
            {copied ? <CheckCheck className="mr-2 h-4 w-4" /> : <Clipboard className="mr-2 h-4 w-4" />}
            {copied ? 'Copied' : 'Copy JSON'}
          </Button>
          <Button variant="ghost" onClick={downloadJson} className="h-9">
            <Download className="mr-2 h-4 w-4" />
            Download JSON
          </Button>
        </div>

        <details className="rounded-lg border border-slate-800 bg-slate-950/70">
          <summary className="cursor-pointer list-none px-3 py-2 text-sm text-slate-300">
            <span className="inline-flex items-center gap-2">
              <FileJson className="h-4 w-4 text-slate-500" />
              Show raw response payload
            </span>
          </summary>
          <pre className="max-h-64 overflow-auto border-t border-slate-800 px-3 py-2 text-xs text-slate-300">
            {rawJson}
          </pre>
        </details>
      </Card>
    </section>
  )
}
