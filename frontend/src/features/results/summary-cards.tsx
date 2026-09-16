import { Badge } from '../../components/ui/badge'

import { Card } from '../../components/ui/card'

import {

  formatConfidence,

  formatNullable,


  getRoutingDecisionDisplayTone,

  getRoutingDecisionLabel,

  getSpecificationCheckLabel,

} from '../../lib/format'

import {

  getCompliantNeedsReviewNote,
  getPresentationConfidenceHelperText,
  getPresentationSpecificationCheckDisplay,
  resolveDecisionConfidence,
} from '../../lib/presentation-safety'

import type { ExtractionResponse } from '../../types/qualiflow'



interface SummaryCardsProps {

  data: ExtractionResponse

  variant?: 'default' | 'detail'

}



function confidenceTone(score: number): 'success' | 'warning' | 'danger' {

  if (score >= 0.85) return 'success'

  if (score >= 0.6) return 'warning'

  return 'danger'

}



export function SummaryCards({ data, variant = 'default' }: SummaryCardsProps) {

  if (variant === 'detail') {

    const metrics = [

      { label: 'Supplier', value: formatNullable(data.supplier_name, 'Unknown') },

      { label: 'Document Type', value: formatNullable(data.document_type, 'Unknown') },

      { label: 'Certificate Date', value: formatNullable(data.certificate_date) },

      { label: 'Certificate No', value: formatNullable(data.certificate_number) },

      { label: 'Order / PO', value: formatNullable(data.order_number) },

      { label: 'Line Items', value: String(data.total_items_detected) },

    ]



    return (

      <section className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">

        {metrics.map((metric) => (

          <Card key={metric.label} className="space-y-1">

            <p className="text-xs uppercase tracking-wider text-slate-500">{metric.label}</p>

            <p className="text-base font-semibold text-slate-100">{metric.value}</p>

          </Card>

        ))}

      </section>

    )

  }



  const specificationCheck = getSpecificationCheckLabel(data)

  const routingDecision = getRoutingDecisionLabel(data)

  const specificationDisplay = getPresentationSpecificationCheckDisplay(data)

  const showCompliantNeedsReviewNote =

    specificationCheck === 'Compliant' && routingDecision === 'Needs human review'

  const traceabilityBlocksCompliance =

    data.traceability_status !== 'VERIFIED' && data.review_reasons?.includes('traceability_unverified')

  const confidenceHelper = getPresentationConfidenceHelperText(data)



  const validationCardValue = (

    <div className="space-y-3">

      <div className="space-y-1">

        <p className="text-xs uppercase tracking-wider text-slate-500">Rule-based compliance check</p>

        <Badge text={specificationDisplay.label} tone={specificationDisplay.tone} />

        {specificationDisplay.reason && (

          <p className="text-xs font-normal leading-relaxed text-slate-400">

            Reason: {specificationDisplay.reason}

          </p>

        )}

      </div>

      <div className="space-y-1">

        <p className="text-xs uppercase tracking-wider text-slate-500">Final routing decision</p>

        <Badge text={routingDecision} tone={getRoutingDecisionDisplayTone(routingDecision)} />

      </div>

      {showCompliantNeedsReviewNote && (

        <p className="text-xs font-normal leading-relaxed text-slate-400">

          {getCompliantNeedsReviewNote(data)}

        </p>

      )}

    </div>

  )



  const cards = [

    { label: 'Supplier Name', value: formatNullable(data.supplier_name, 'Unknown') },

    { label: 'Document Type', value: formatNullable(data.document_type, 'Unknown') },

    { label: 'Certificate Date', value: formatNullable(data.certificate_date) },

    { label: 'Total Items Detected', value: String(data.total_items_detected) },

    {

      label: 'Decision Confidence',

      value: (

        <div className="space-y-2">

          <Badge
            text={formatConfidence(resolveDecisionConfidence(data))}
            tone={confidenceTone(resolveDecisionConfidence(data))}
          />

          {confidenceHelper && (

            <p className="text-xs font-normal leading-relaxed text-slate-400">{confidenceHelper}</p>

          )}

        </div>

      ),

    },

    {

      label: 'Validation & Routing',

      value: validationCardValue,

    },

  ]



  return (

    <section className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">

      {cards.map((card) => (

        <Card key={card.label} className="space-y-2">

          <p className="text-xs uppercase tracking-wider text-slate-400">{card.label}</p>

          <div className="text-base font-semibold text-slate-100">{card.value}</div>

        </Card>

      ))}

      {traceabilityBlocksCompliance && (

        <Card className="space-y-2 md:col-span-2 xl:col-span-3">

          <Badge

            text="Mechanical compliance passed, but traceability identifiers require human verification."

            tone="warning"

          />

        </Card>

      )}

    </section>

  )

}


