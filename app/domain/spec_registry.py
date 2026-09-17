"""Spec resolution layer.

Given a :class:`~app.domain.grade_registry.GradeResolution` for a grade, this
module returns a :class:`SpecResolution` that tells the validator what the
compliance thresholds are, or explicitly marks the spec as unresolved (for
example stainless product-form-dependent grades).

The validator never invents numbers. If a grade resolves to a family but no
spec is declared, :func:`resolve_spec` returns
``SpecResolution(status='unresolved', ...)`` and the validator emits a review
token instead of a non-compliant verdict.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from app.domain.grade_registry import GradeResolution

SpecStatus = Literal[
    "resolved",
    "unresolved",
    "unknown_grade",
    "ambiguous_grade",
    "unsupported_spec_family",
    "empty",
]


@dataclass(frozen=True)
class MaterialSpec:
    """Compliance thresholds for a single canonical grade family."""

    canonical: str
    min_yield_mpa: float
    min_tensile_mpa: float
    max_tensile_mpa: float
    min_elongation_pct: float


# Carbon-steel specs reuse the existing QualiFlow thesis baseline. The numbers
# match EN 10025-2 minimum yield / elongation headlines; tensile bands are
# slightly wider than the standard's headline ranges because the thesis
# baseline absorbs supplier reporting variance. Do not tighten these without
# updating the validator-baseline comparison.
_CARBON_SPECS: dict[str, MaterialSpec] = {
    "S235JR": MaterialSpec("S235JR", 235.0, 360.0, 510.0, 26.0),
    "S275JR": MaterialSpec("S275JR", 275.0, 410.0, 560.0, 23.0),
    "S355JR": MaterialSpec("S355JR", 355.0, 470.0, 630.0, 22.0),
    "S355J2": MaterialSpec("S355J2", 355.0, 470.0, 630.0, 22.0),
    "S355J2+N": MaterialSpec("S355J2+N", 355.0, 470.0, 630.0, 22.0),
    "API 5L X65": MaterialSpec("API 5L X65", 448.0, 531.0, 758.0, 18.0),
    "SG2": MaterialSpec("SG2", 420.0, 500.0, 640.0, 22.0),
    # EN 10255-style S195 pipe baseline. Keep the wider 350-620 tensile band
    # aligned with the current project convention for supplier variance.
    "S195": MaterialSpec("S195", 195.0, 350.0, 620.0, 20.0),
    # BS 4449 / EN 10080 B500B rebar minima (elongation optional when absent).
    "B500B": MaterialSpec("B500B", 500.0, 540.0, 2000.0, 0.0),
    # Demo-supported wire-rod grade: permissive sanity bands only.
    "SRCDRW02": MaterialSpec("SRCDRW02", 150.0, 250.0, 900.0, 0.0),
    # Narrow thesis support for the observed 321/321H stainless MTCs. These
    # minima are intentionally conservative and only unlock deterministic
    # validation for the declared aliases below.
    "1.4541": MaterialSpec("1.4541", 205.0, 515.0, 750.0, 35.0),
    "321": MaterialSpec("321", 205.0, 515.0, 750.0, 35.0),
    "1.4878": MaterialSpec("1.4878", 205.0, 515.0, 750.0, 35.0),
    "321H": MaterialSpec("321H", 205.0, 515.0, 750.0, 35.0),
}


# Stainless austenitic grades are intentionally routed as unsupported in the
# prototype validator: they require product-form-aware rule sets that are
# outside the current deterministic rule baseline.
_INTENTIONALLY_UNRESOLVED: set[str] = {
    "1.4301",
    "1.4307",
    "1.4401",
    "1.4404",
    "304",
    "304L",
    "316",
    "316L",
    "317",
    "317L",
    "347",
    "347H",
    "TP304",
    "TP304L",
    "TP316",
    "TP316L",
    "TP317",
    "TP317L",
    "TP321",
    "TP321H",
    "TP347",
    "TP347H",
    "316Ti",
    "1.4550",
    "1.4571",
}


@dataclass(frozen=True)
class SpecResolution:
    """Outcome of mapping a :class:`GradeResolution` to a material spec."""

    status: SpecStatus
    spec: MaterialSpec | None
    canonical: str | None
    candidates: tuple[str, ...]
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "canonical": self.canonical,
            "candidates": list(self.candidates),
            "reason": self.reason,
            "spec": (
                {
                    "canonical": self.spec.canonical,
                    "min_yield_mpa": self.spec.min_yield_mpa,
                    "min_tensile_mpa": self.spec.min_tensile_mpa,
                    "max_tensile_mpa": self.spec.max_tensile_mpa,
                    "min_elongation_pct": self.spec.min_elongation_pct,
                }
                if self.spec is not None
                else None
            ),
        }


def resolve_spec(grade: GradeResolution) -> SpecResolution:
    """Resolve a :class:`GradeResolution` to a material spec."""

    if grade.status == "empty":
        return SpecResolution(
            status="empty",
            spec=None,
            canonical=None,
            candidates=(),
            reason="no grade supplied",
        )

    if grade.status == "unknown":
        return SpecResolution(
            status="unknown_grade",
            spec=None,
            canonical=None,
            candidates=(),
            reason="grade not in registry",
        )

    if grade.status == "ambiguous":
        return SpecResolution(
            status="ambiguous_grade",
            spec=None,
            canonical=None,
            candidates=grade.candidates,
            reason="grade resolution was ambiguous across families",
        )

    # Both ``resolved`` and ``resolved_dual``: try the preferred canonical.
    preferred_candidates = list(grade.candidates) or ([grade.canonical] if grade.canonical else [])
    for candidate in preferred_candidates:
        spec = _CARBON_SPECS.get(candidate)
        if spec is not None:
            return SpecResolution(
                status="resolved",
                spec=spec,
                canonical=candidate,
                candidates=tuple(preferred_candidates),
                reason="carbon-steel spec resolved",
            )

    # Known family, but spec intentionally not declared (e.g. stainless).
    if any(candidate in _INTENTIONALLY_UNRESOLVED for candidate in preferred_candidates):
        return SpecResolution(
            status="unsupported_spec_family",
            spec=None,
            canonical=grade.canonical,
            candidates=tuple(preferred_candidates),
            reason="grade family recognised but deterministic family rules are not declared",
        )

    return SpecResolution(
        status="unresolved",
        spec=None,
        canonical=grade.canonical,
        candidates=tuple(preferred_candidates),
        reason="no spec declared for this grade family",
    )


def known_spec_grades() -> list[str]:
    """Sorted list of canonical grades with a declared spec (carbon-steel)."""

    return sorted(_CARBON_SPECS.keys())


def get_spec(canonical: str) -> MaterialSpec | None:
    return _CARBON_SPECS.get(canonical)


def has_deterministic_spec(canonical: str) -> bool:
    """Return True when *canonical* has an explicit threshold entry."""

    return canonical in _CARBON_SPECS


def _normalize_pair_part(value: str) -> str:
    return " ".join(value.upper().split())


_CLASSIFICATION_SPECS: dict[tuple[str, str], MaterialSpec] = {}


def register_classification_spec(standard: str, classification: str, spec: MaterialSpec) -> None:
    """Register a supported ``(standard, classification)`` spec pair.

    Production stays empty unless a domain pair is explicitly declared.
    Tests may inject pairs without inventing a grade.
    """

    _CLASSIFICATION_SPECS[(_normalize_pair_part(standard), _normalize_pair_part(classification))] = spec


def clear_classification_specs() -> None:
    _CLASSIFICATION_SPECS.clear()


def resolve_spec_from_standard_classification(
    standards: list[str] | None,
    classifications: list[str] | None,
    *,
    registry: dict[tuple[str, str], MaterialSpec] | None = None,
) -> SpecResolution:
    """Resolve mechanical thresholds from a supported standard/classification pair."""

    table = registry if registry is not None else _CLASSIFICATION_SPECS
    if not table:
        return SpecResolution(
            status="empty",
            spec=None,
            canonical=None,
            candidates=(),
            reason="no supported standard/classification pair declared",
        )

    standard_values = [str(item).strip() for item in (standards or []) if str(item).strip()]
    classification_values = [str(item).strip() for item in (classifications or []) if str(item).strip()]
    if not standard_values or not classification_values:
        return SpecResolution(
            status="empty",
            spec=None,
            canonical=None,
            candidates=(),
            reason="missing standard or classification evidence",
        )

    matches: list[tuple[str, str, MaterialSpec]] = []
    for standard in standard_values:
        std_key = _normalize_pair_part(standard)
        for classification in classification_values:
            cls_key = _normalize_pair_part(classification)
            spec = table.get((std_key, cls_key))
            if spec is not None:
                matches.append((standard, classification, spec))
                continue
            for registered_std, registered_cls in table:
                if registered_cls == cls_key and (
                    std_key == registered_std or registered_std in std_key or std_key in registered_std
                ):
                    matches.append((standard, classification, table[(registered_std, registered_cls)]))
                    break

    unique_specs = {item[2].canonical: item[2] for item in matches}
    if len(unique_specs) == 1:
        spec = next(iter(unique_specs.values()))
        return SpecResolution(
            status="resolved",
            spec=spec,
            canonical=spec.canonical,
            candidates=tuple(unique_specs),
            reason="standard/classification pair resolved",
        )
    if len(unique_specs) > 1:
        return SpecResolution(
            status="ambiguous_grade",
            spec=None,
            canonical=None,
            candidates=tuple(unique_specs),
            reason="multiple standard/classification specs matched",
        )
    return SpecResolution(
        status="empty",
        spec=None,
        canonical=None,
        candidates=(),
        reason="no supported standard/classification pair declared",
    )


__all__ = [
    "MaterialSpec",
    "SpecResolution",
    "clear_classification_specs",
    "register_classification_spec",
    "resolve_spec",
    "resolve_spec_from_standard_classification",
    "known_spec_grades",
    "get_spec",
    "has_deterministic_spec",
]
