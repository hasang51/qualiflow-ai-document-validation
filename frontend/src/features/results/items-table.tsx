import {
  type ColumnDef,
  flexRender,
  getCoreRowModel,
  getFilteredRowModel,
  getSortedRowModel,
  useReactTable,
  type SortingState,
} from '@tanstack/react-table'
import { ChevronDown, ChevronUp, Search } from 'lucide-react'
import { Fragment, useMemo, useState } from 'react'
import { Badge } from '../../components/ui/badge'
import { Card } from '../../components/ui/card'
import { Input } from '../../components/ui/input'
import { renderCriticalIdentifier } from '../../lib/critical-identifiers'
import { formatMpaDisplay, formatNullable, formatNumber, MISSING_VALUE } from '../../lib/format'
import { SIZE_WEIGHT_COLUMN_HEADER } from '../../lib/presentation-labels'
import {
  isItemMechanicallyIncomplete,
  resolveItemRefCellValue,
  resolveItemRefColumnHeader,
} from '../../lib/presentation-safety'
import { dedupeReasons, formatDisplayReviewReason } from '../../lib/presentation-safety'
import type { ExtractedItem, GradeResolutionPayload, TraceabilityStatus, ValidationOutcome } from '../../types/qualiflow'

const ALTERNATIVE_CLASSIFICATION_REASON = 'row_shape:alternative_classification_rows_collapsed'
const ALTERNATIVE_CLASSIFICATION_NOTE =
  'This certificate lists multiple classification conditions. The table shows the primary extracted row; alternate classifications should be reviewed in the source document.'

const DEFAULT_EMPTY_ITEMS_MESSAGE = 'No line items were detected in this document.'

interface ItemsTableProps {
  items: ExtractedItem[]
  reviewReasons?: string[]
  explanation?: Record<string, unknown> | null
  emptyMessage?: string
}

function isEmptyItemId(value: string | null): boolean {
  if (value === null || value === undefined) return true
  const trimmed = value.trim()
  return trimmed === '' || trimmed === MISSING_VALUE || trimmed === '—'
}

interface RowShape {
  index: number
  heatNo: string | null
  itemId: string | null
  productName: string | null
  grade: string | null
  weightOrLength: string | null
  dimensions: string | null
  standards: string[]
  yieldMpa: number | null
  tensileMpa: number | null
  elongation: number | null
  isCompliant: boolean | null
  outcome: ValidationOutcome | null
  needsReview: boolean
  traceabilityStatus: TraceabilityStatus | null
  traceabilityConfidence: number | null
  deviations: string[]
  gradeResolution: GradeResolutionPayload | null
  gradeProvenance: string | null
}

function complianceText(
  value: boolean | null,
  outcome: ValidationOutcome | string | null,
  needsReview = false,
  traceabilityStatus?: TraceabilityStatus | null,
  mechanicallyIncomplete = false,
): { text: string; tone: 'success' | 'danger' | 'warning' | 'info' } {
  if (mechanicallyIncomplete) {
    return { text: 'Needs review', tone: 'warning' }
  }
  if ((outcome === 'COMPLIANT' || value === true) && traceabilityStatus !== 'VERIFIED') {
    return { text: 'Needs review', tone: 'warning' }
  }
  if (needsReview) return { text: 'Needs review', tone: 'warning' }

  // Outcome takes precedence over the tri-state boolean when available so
  // unresolved / unknown / ambiguous grades land in the warning tone rather
  // than the false-negative "Non-compliant" red badge.
  switch (outcome) {
    case 'COMPLIANT':
    case 'RESOLVED_COMPLIANT':
      return { text: 'Compliant', tone: 'success' }
    case 'NON_COMPLIANT':
    case 'RESOLVED_NON_COMPLIANT':
      return { text: 'Non-compliant', tone: 'danger' }
    case 'NEEDS_REVIEW':
      return { text: 'Needs review', tone: 'warning' }
    case 'UNRESOLVED_SPEC':
      return { text: 'Spec unresolved - review', tone: 'warning' }
    case 'UNSUPPORTED_SPEC_FAMILY':
      return { text: 'Unsupported family - review', tone: 'warning' }
    case 'EXPLICIT_UNMAPPED_GRADE':
      return { text: 'Explicit grade not mapped', tone: 'warning' }
    case 'MISSING_CRITICAL_FIELD_GRADE':
      return { text: 'Missing grade', tone: 'warning' }
    case 'NOT_VALIDATED':
    case 'NOT_APPLICABLE':
      return { text: 'Not validated', tone: 'info' }
    case 'EXTRACTION_UNCERTAIN':
    case 'UNKNOWN_GRADE':
    case 'AMBIGUOUS_GRADE':
      return { text: 'Needs review', tone: 'warning' }
    default:
      if (value === true) return { text: 'Compliant', tone: 'success' }
      if (value === false) return { text: 'Non-compliant', tone: 'danger' }
      return { text: 'Not validated', tone: 'warning' }
  }
}

export function ItemsTable({
  items,
  reviewReasons = [],
  explanation = null,
  emptyMessage = DEFAULT_EMPTY_ITEMS_MESSAGE,
}: ItemsTableProps) {
  const [sorting, setSorting] = useState<SortingState>([])
  const [globalFilter, setGlobalFilter] = useState('')
  const [expandedRows, setExpandedRows] = useState<Record<string, boolean>>({})

  const rows = useMemo<RowShape[]>(
    () =>
      items.map((item, index) => {
        const renderedItemId = renderCriticalIdentifier(
          item,
          ['item_id', 'pipe_id', 'pipe_coil_id'],
          MISSING_VALUE,
          { allowCrossFieldFallback: false, allowTraceabilityShortcut: false },
        )
        return {
        index,
        heatNo: renderCriticalIdentifier(item, [
          'traceability_identifier_value',
          'heat_number',
          'batch_number',
          'colata_number',
          'lot_number',
          'cast_number',
          'charge_number',
        ]),
        itemId: resolveItemRefCellValue(
          item,
          index,
          renderedItemId === MISSING_VALUE ? null : renderedItemId,
          explanation,
          MISSING_VALUE,
        ),
        productName: item.product_name ?? null,
        grade: item.grade,
        weightOrLength: item.weight_or_length,
        dimensions: item.dimensions ?? null,
        standards: item.standards ?? [],
        yieldMpa: item.mechanical_properties?.yield_strength_mpa ?? null,
        tensileMpa: item.mechanical_properties?.tensile_strength_mpa ?? null,
        elongation: item.mechanical_properties?.elongation_percentage ?? null,
        isCompliant: item.validation?.is_compliant ?? null,
        outcome: (item.validation?.outcome as ValidationOutcome | null | undefined) ?? null,
        needsReview: item.needs_review === true,
        traceabilityStatus: item.traceability_status ?? null,
        traceabilityConfidence: item.traceability_confidence ?? null,
        deviations: item.validation?.deviations ?? [],
        gradeResolution: item.grade_resolution ?? null,
        gradeProvenance: item.grade_provenance ?? null,
        }
      }),
    [items, explanation],
  )

  const showItemIdColumn = useMemo(
    () => rows.some((row) => !isEmptyItemId(row.itemId)),
    [rows],
  )

  const showDimensionsColumn = useMemo(
    () => rows.some((row) => Boolean(row.dimensions && row.dimensions.trim())),
    [rows],
  )

  const showProductColumn = useMemo(
    () => rows.some((row) => Boolean(row.productName && row.productName.trim())),
    [rows],
  )

  const showAlternativeClassificationNote = useMemo(
    () => reviewReasons.includes(ALTERNATIVE_CLASSIFICATION_REASON),
    [reviewReasons],
  )

  const itemRefColumnHeader = useMemo(
    () => resolveItemRefColumnHeader(items, explanation),
    [items, explanation],
  )

  const columns = useMemo<ColumnDef<RowShape>[]>(
    () => {
      const baseColumns: ColumnDef<RowShape>[] = [
      {
        accessorKey: 'heatNo',
        header: 'Traceability ID',
        cell: ({ row }) => formatNullable(row.original.heatNo),
      },
      ...(showItemIdColumn
        ? [
            {
              accessorKey: 'itemId',
              header: itemRefColumnHeader,
              cell: ({ row }) => formatNullable(row.original.itemId),
            } as ColumnDef<RowShape>,
          ]
        : []),
      ...(showProductColumn
        ? [
            {
              accessorKey: 'productName',
              header: 'Product',
              cell: ({ row }) => formatNullable(row.original.productName),
            } as ColumnDef<RowShape>,
          ]
        : []),
      {
        accessorKey: 'grade',
        header: 'Grade',
        cell: ({ row }) => formatNullable(row.original.grade),
      },
      {
        accessorKey: 'weightOrLength',
        header: SIZE_WEIGHT_COLUMN_HEADER,
        cell: ({ row }) => formatNullable(row.original.weightOrLength),
      },
      ...(showDimensionsColumn
        ? [
            {
              accessorKey: 'dimensions',
              header: 'Dimensions',
              cell: ({ row }) => formatNullable(row.original.dimensions),
            } as ColumnDef<RowShape>,
          ]
        : []),
      {
        accessorKey: 'yieldMpa',
        header: 'Yield',
        cell: ({ row }) => formatMpaDisplay(row.original.yieldMpa),
      },
      {
        accessorKey: 'tensileMpa',
        header: 'Tensile',
        cell: ({ row }) => formatMpaDisplay(row.original.tensileMpa),
      },
      {
        id: 'compliance',
        header: 'Row Status',
        cell: ({ row }) => {
          const item = items[row.original.index]
          const state = complianceText(
            row.original.isCompliant,
            row.original.outcome,
            row.original.needsReview,
            row.original.traceabilityStatus,
            isItemMechanicallyIncomplete(item, reviewReasons),
          )
          return <Badge text={state.text} tone={state.tone} />
        },
      },
      {
        id: 'details',
        enableSorting: false,
        header: '',
        cell: ({ row }) => {
          const key = String(row.original.index)
          const expanded = expandedRows[key] ?? false
          return (
            <button
              type="button"
              className="rounded border border-slate-700 px-2 py-1 text-xs text-slate-200 hover:bg-slate-800"
              onClick={() =>
                setExpandedRows((current) => ({
                  ...current,
                  [key]: !expanded,
                }))
              }
            >
              {expanded ? 'Hide' : 'Details'}
            </button>
          )
        },
      },
    ]

      return baseColumns
    },
    [expandedRows, itemRefColumnHeader, items, reviewReasons, showDimensionsColumn, showItemIdColumn, showProductColumn],
  )

  // eslint-disable-next-line react-hooks/incompatible-library
  const table = useReactTable({
    data: rows,
    columns,
    state: { sorting, globalFilter },
    onSortingChange: setSorting,
    onGlobalFilterChange: setGlobalFilter,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    getFilteredRowModel: getFilteredRowModel(),
    globalFilterFn: 'includesString',
  })

  if (items.length === 0) {
    return (
      <Card>
        <p className="text-sm text-slate-300">{emptyMessage}</p>
      </Card>
    )
  }

  return (
    <Card className="space-y-4 p-0">
      <div className="space-y-3 border-b border-slate-800 p-4">
        <div className="flex items-center justify-between gap-4">
          <div className="relative w-full max-w-sm">
            <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-500" />
            <Input
              value={globalFilter}
              onChange={(event) => setGlobalFilter(event.target.value)}
              placeholder="Search traceability ID, source ref, product, grade..."
              className="pl-9"
            />
          </div>
          <p className="text-xs text-slate-400">{table.getRowModel().rows.length} rows</p>
        </div>
        {showAlternativeClassificationNote && (
          <p className="text-xs leading-relaxed text-slate-500">{ALTERNATIVE_CLASSIFICATION_NOTE}</p>
        )}
      </div>

      <div className="max-h-[520px] overflow-auto">
        <table className="min-w-full border-collapse text-sm">
          <thead className="sticky top-0 z-10 bg-slate-950/95 backdrop-blur">
            {table.getHeaderGroups().map((headerGroup) => (
              <tr key={headerGroup.id} className="border-b border-slate-800">
                {headerGroup.headers.map((header) => (
                  <th key={header.id} className="px-4 py-3 text-left font-semibold text-slate-300">
                    {header.isPlaceholder ? null : (
                      <button
                        type="button"
                        className="inline-flex items-center gap-1 disabled:pointer-events-none disabled:opacity-100"
                        disabled={!header.column.getCanSort()}
                        onClick={header.column.getToggleSortingHandler()}
                      >
                        {flexRender(header.column.columnDef.header, header.getContext())}
                        {{
                          asc: <ChevronUp className="h-3.5 w-3.5" />,
                          desc: <ChevronDown className="h-3.5 w-3.5" />,
                        }[header.column.getIsSorted() as string] ?? null}
                      </button>
                    )}
                  </th>
                ))}
              </tr>
            ))}
          </thead>
          <tbody>
            {table.getRowModel().rows.map((row) => {
              const key = String(row.original.index)
              const expanded = expandedRows[key] ?? false

              return (
                <Fragment key={row.id}>
                  <tr key={row.id} className="border-b border-slate-900/80 hover:bg-slate-900/70">
                    {row.getVisibleCells().map((cell) => (
                      <td key={cell.id} className="px-4 py-3 text-slate-200">
                        {flexRender(cell.column.columnDef.cell, cell.getContext())}
                      </td>
                    ))}
                  </tr>
                  {expanded && (
                    <tr className="border-b border-slate-900/80 bg-slate-950/70">
                      <td colSpan={columns.length} className="px-4 py-3">
                        <div className="space-y-2 text-sm">
                          <p className="text-slate-300">
                            <span className="mr-2 text-slate-500">Elongation:</span>
                            {formatNumber(row.original.elongation)}
                          </p>
                          {row.original.productName && (
                            <p className="text-slate-300">
                              <span className="mr-2 text-slate-500">Product:</span>
                              {row.original.productName}
                            </p>
                          )}
                          <p className="text-slate-300">
                            <span className="mr-2 text-slate-500">Grade:</span>
                            {formatNullable(row.original.grade)}
                          </p>
                          {row.original.outcome && (
                            <p className="text-slate-300">
                              <span className="mr-2 text-slate-500">Validation outcome:</span>
                              {row.original.outcome}
                            </p>
                          )}
                          {items[row.original.index]?.certificate_number && (
                            <p className="text-slate-300">
                              <span className="mr-2 text-slate-500">Certificate:</span>
                              {items[row.original.index].certificate_number}
                            </p>
                          )}
                          {items[row.original.index]?.order_number && (
                            <p className="text-slate-300">
                              <span className="mr-2 text-slate-500">Order / PO:</span>
                              {items[row.original.index].order_number}
                            </p>
                          )}
                          {items[row.original.index]?.dimensions && (
                            <p className="text-slate-300">
                              <span className="mr-2 text-slate-500">Dimensions:</span>
                              {items[row.original.index].dimensions}
                            </p>
                          )}
                          {items[row.original.index]?.standards && items[row.original.index].standards!.length > 0 && (
                            <p className="text-slate-300">
                              <span className="mr-2 text-slate-500">Standards:</span>
                              {items[row.original.index].standards!.join(', ')}
                            </p>
                          )}
                          {items[row.original.index]?.chemical_composition && (
                            <p className="text-slate-300">
                              <span className="mr-2 text-slate-500">Chemistry:</span>
                              {Object.entries(items[row.original.index].chemical_composition ?? {})
                                .filter(([, value]) => value !== null && value !== undefined)
                                .map(([element, value]) => `${element} ${value}`)
                                .join(', ') || MISSING_VALUE}
                            </p>
                          )}
                          {row.original.traceabilityStatus && (
                            <p className="text-slate-300">
                              <span className="mr-2 text-slate-500">Traceability:</span>
                              {row.original.traceabilityStatus}
                              {row.original.traceabilityConfidence !== null
                                ? ` (${Math.round(row.original.traceabilityConfidence * 100)}%)`
                                : ''}
                            </p>
                          )}
                          {row.original.gradeResolution?.candidates && row.original.gradeResolution.candidates.length > 0 && (
                            <p className="text-slate-300">
                              <span className="mr-2 text-slate-500">Grade candidates:</span>
                              {row.original.gradeResolution.candidates.join(', ')}
                              {row.original.gradeProvenance ? ` (${row.original.gradeProvenance})` : ''}
                            </p>
                          )}
                          <div className="text-slate-300">
                            <span className="mr-2 text-slate-500">Validation deviations:</span>
                            {row.original.deviations.length > 0 ? (
                              <ul className="mt-2 list-disc space-y-1 pl-5">
                                {dedupeReasons(row.original.deviations).map((deviation, idx) => (
                                  <li key={`${key}-${idx}`}>{formatDisplayReviewReason(deviation)}</li>
                                ))}
                              </ul>
                            ) : (
                              <span>{MISSING_VALUE}</span>
                            )}
                          </div>
                        </div>
                      </td>
                    </tr>
                  )}
                </Fragment>
              )
            })}
          </tbody>
        </table>
      </div>
    </Card>
  )
}
