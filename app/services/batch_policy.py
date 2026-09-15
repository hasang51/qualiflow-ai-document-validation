"""Centralised cost-control policy for batch extraction runs.

All token-budget, rate-limit, and page-cap decisions flow through
:class:`BatchRunPolicy`. Batch scripts read a single policy object rather
than spreading magic constants across many files.

Default values are set for a **pilot study** phase where the Bedrock token
budget is limited and estimated API cost should stay well under a few dollars.

Usage::

    from app.services.batch_policy import BatchRunPolicy, DEFAULT_PILOT_POLICY

    policy = DEFAULT_PILOT_POLICY          # or BatchRunPolicy(**overrides)

    # Check whether to skip a document before any API call:
    skip, reason = policy.should_skip_document(quality_class, page_count)

    # Decide pages to send:
    max_pages = policy.max_pages_for_quality(quality_class)

    # Accumulate spend:
    policy.record_usage(input_tokens=120, output_tokens=40)

    # After each doc, check whether budget is exhausted:
    ok, reason = policy.budget_ok()
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

# Approximate estimated cost per 1 000 tokens for Gemma 4 26B-A4B in eu-central-1
# Standard (not billing-grade). Conservative enough to stop a pilot run safely.
_COST_PER_1K_INPUT_USD = 0.00016   # $0.16 / 1M input tokens
_COST_PER_1K_OUTPUT_USD = 0.00048  # $0.48 / 1M output tokens
# Image token estimates vary by Gemma vision tokenization; keep a conservative
# ~800–1 100 tokens per packed page image.
_ESTIMATED_TOKENS_PER_IMAGE = 1000


@dataclass
class BatchRunPolicy:
    """All tunable knobs for a cost-controlled batch run.

    Attributes
    ----------
    max_new_live_docs:
        Maximum number of documents for which new Bedrock calls are issued in
        this run. Already-cached per-document JSONs (``--resume``) never count.
    max_pages_per_doc:
        Hard cap on pages sent to the LLM for *any* document. The page
        selector may choose fewer based on document quality.
    max_pages_noisy_scan:
        Override page cap specifically for ``noisy_scan`` docs.
        Allowed to be slightly higher than ``max_pages_per_doc`` because
        degraded docs sometimes need a second page for the table.
    skip_severe_scan:
        If ``True``, documents whose ``blur_score < severe_blur_threshold``
        are skipped entirely (no API call, recorded as ``SKIPPED_SEVERE_SCAN``).
    severe_blur_threshold:
        Blur score (Laplacian variance) below which a doc is considered
        too degraded to be worth the token spend in this phase.
    inter_doc_sleep_s:
        Seconds to pause between consecutive live API calls.
    stop_on_estimated_budget_usd:
        Halt the batch (but persist already-written outputs) when cumulative
        estimated spend reaches this threshold.
    stop_on_second_rate_limit:
        Halt the batch after two consecutive 429 / rate-limit events so we
        do not burn the retry budget on an exhausted org quota.
    use_resume:
        If ``True``, skip documents whose per-document JSON already exists.
    """

    max_new_live_docs: int = 5
    max_pages_per_doc: int = 2
    max_pages_noisy_scan: int = 3
    skip_severe_scan: bool = True
    severe_blur_threshold: float = 30.0   # below this = almost certainly unreadable
    inter_doc_sleep_s: float = 45.0
    stop_on_estimated_budget_usd: float = 2.00
    stop_on_second_rate_limit: bool = True
    use_resume: bool = True

    # Runtime accumulators — not config inputs.
    _total_input_tokens: int = field(default=0, init=False, repr=False)
    _total_output_tokens: int = field(default=0, init=False, repr=False)
    _total_new_live_docs: int = field(default=0, init=False, repr=False)
    _rate_limit_events: int = field(default=0, init=False, repr=False)

    # ------------------------------------------------------------------ helpers

    def max_pages_for_quality(self, quality_class: str) -> int:
        """Return the page cap for a given quality class."""
        if quality_class == "noisy_scan":
            return self.max_pages_noisy_scan
        return self.max_pages_per_doc

    def should_skip_document(
        self,
        quality_class: str,
        blur_score: float | None,
    ) -> tuple[bool, str]:
        """Return ``(should_skip, reason)``."""
        if self._total_new_live_docs >= self.max_new_live_docs:
            return True, f"max_new_live_docs={self.max_new_live_docs} reached"
        if self.skip_severe_scan and quality_class == "noisy_scan":
            if blur_score is not None and blur_score < self.severe_blur_threshold:
                return True, f"severe_scan: blur_score={blur_score:.1f} < {self.severe_blur_threshold}"
        return False, ""

    def budget_ok(self) -> tuple[bool, str]:
        """Return ``(within_budget, reason)``."""
        estimated = self.estimated_spend_usd()
        if estimated >= self.stop_on_estimated_budget_usd:
            return False, (
                f"estimated_spend_usd={estimated:.4f} >= "
                f"stop_on_estimated_budget_usd={self.stop_on_estimated_budget_usd}"
            )
        return True, ""

    def record_rate_limit(self) -> tuple[bool, str]:
        """Record a rate-limit event. Return ``(should_stop, reason)``."""
        self._rate_limit_events += 1
        if self.stop_on_second_rate_limit and self._rate_limit_events >= 2:
            return True, f"rate_limit_events={self._rate_limit_events} >= 2"
        return False, ""

    def record_usage(self, *, input_tokens: int, output_tokens: int) -> None:
        self._total_input_tokens += input_tokens
        self._total_output_tokens += output_tokens

    def record_new_live_doc(self) -> None:
        self._total_new_live_docs += 1

    def estimated_spend_usd(self) -> float:
        cost = (
            self._total_input_tokens / 1000.0 * _COST_PER_1K_INPUT_USD
            + self._total_output_tokens / 1000.0 * _COST_PER_1K_OUTPUT_USD
        )
        return round(cost, 6)

    def usage_summary(self) -> dict:
        return {
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
            "total_new_live_docs": self._total_new_live_docs,
            "rate_limit_events": self._rate_limit_events,
            "estimated_spend_usd": self.estimated_spend_usd(),
        }

    @classmethod
    def from_env(cls) -> "BatchRunPolicy":
        """Build a policy from environment variables (all optional)."""
        return cls(
            max_new_live_docs=int(os.getenv("QUALIFLOW_MAX_LIVE_DOCS", "5")),
            max_pages_per_doc=int(os.getenv("QUALIFLOW_MAX_PAGES_PER_DOC", "2")),
            max_pages_noisy_scan=int(os.getenv("QUALIFLOW_MAX_PAGES_NOISY_SCAN", "3")),
            skip_severe_scan=os.getenv("QUALIFLOW_SKIP_SEVERE_SCAN", "1").lower() in {
                "1", "true", "yes"
            },
            severe_blur_threshold=float(os.getenv("QUALIFLOW_SEVERE_BLUR_THRESHOLD", "30.0")),
            inter_doc_sleep_s=float(os.getenv("QUALIFLOW_BATCH_SLEEP_S", "45.0")),
            stop_on_estimated_budget_usd=float(
                os.getenv("QUALIFLOW_BUDGET_USD", "2.00")
            ),
            stop_on_second_rate_limit=os.getenv(
                "QUALIFLOW_STOP_ON_SECOND_RATE_LIMIT", "1"
            ).lower()
            in {"1", "true", "yes"},
            use_resume=os.getenv("QUALIFLOW_RESUME", "1").lower() in {"1", "true", "yes"},
        )


# Convenience singleton for import by scripts.
DEFAULT_PILOT_POLICY = BatchRunPolicy()
