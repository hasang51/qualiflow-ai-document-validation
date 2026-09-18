"""Deterministic document-level profiler.

Produces a coarse quality classification for a PDF so the extraction router can
pick exactly one runtime strategy. The profiler is intentionally cheap:

- text layer detection uses ``pypdf`` (no OCR, no ML)
- blur / noise proxies reuse the cell-free laplacian + median-residual
  heuristics already used by the per-page preprocessing pipeline
  (``compute_blur_score`` and ``estimate_noise`` in
  :mod:`app.services.image_preprocessing`)
- rasterization is capped at the first few pages only, so a 50-page scan stays
  fast even though every page still gets text-layer inspection

The output ``DocumentProfile`` is a plain dataclass so it serialises trivially
into manifests and batch-run artefacts.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pdf2image import convert_from_path
from PIL import Image

from app.config import settings
from app.services.image_preprocessing import compute_blur_score, estimate_noise

logger = logging.getLogger("qualiflow.profiler")

QualityClass = Literal["digital_clean", "scan_clean", "noisy_scan", "severe_scan"]

DIGITAL_TEXT_DENSITY_MIN = 0.02
"""Minimum mean chars-per-page ratio (normalised) treated as a real text layer."""

BLUR_DEGRADED_THRESHOLD = 80.0
"""Laplacian variance below this is "high blur" (matches preprocessing_strategy)."""

NOISE_DEGRADED_THRESHOLD = 25.0
"""Median-residual stdev above this is "severe noise" (matches preprocessing_strategy)."""

BLUR_MODERATE_THRESHOLD = 150.0
NOISE_MODERATE_THRESHOLD = 14.0
BLUR_SEVERE_THRESHOLD = 25.0
NOISE_SEVERE_THRESHOLD = 45.0

PROFILE_MAX_SAMPLE_PAGES = 2
"""Only rasterise the first ``N`` pages for blur/noise statistics."""

_PROFILE_SMART_RESIZE_MAX_EDGE = 3000
"""Matches :func:`app.services.preprocessing._pil_smart_resize` so blur scores align."""

_LARGE_RASTER_MIN_LONG_EDGE = 1800
"""Longest pixel edge for an embedded image treated as a page-sized scan."""

_LARGE_RASTER_MIN_AREA = 4_000_000
"""Minimum pixel area for a page-sized raster (roughly A4 at ~200 dpi)."""

_LARGE_RASTER_MIN_BYTES = 400_000
"""Fallback when pixel dimensions are unavailable (logos/stamps are much smaller)."""


def _smart_resize(image: Image.Image, max_edge: int = _PROFILE_SMART_RESIZE_MAX_EDGE) -> Image.Image:
    width, height = image.size
    long_edge = max(width, height)
    if long_edge <= max_edge:
        return image
    scale = max_edge / long_edge
    return image.resize(
        (max(1, int(width * scale)), max(1, int(height * scale))),
        Image.Resampling.LANCZOS,
    )


@dataclass
class DocumentProfile:
    document_id: str
    filename: str
    page_count: int
    has_text_layer: bool
    text_density: float
    blur_score: float
    noise_score: float
    table_presence_hint: bool
    quality_class: QualityClass
    reasons: list[str] = field(default_factory=list)
    profile_confidence: float = 0.0
    metrics_summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _image_pixel_size(img: Any) -> tuple[int | None, int | None]:
    """Best-effort width/height for a pypdf embedded image."""

    try:
        pil = img.image
        if pil is not None:
            return int(pil.size[0]), int(pil.size[1])
    except Exception:
        pass

    try:
        ref = img.indirect_reference
        obj = ref.get_object() if hasattr(ref, "get_object") else ref
        width = obj.get("/Width")
        height = obj.get("/Height")
        if width is not None and height is not None:
            return int(width), int(height)
    except Exception:
        pass

    return None, None


def _is_large_raster_image(img: Any) -> bool:
    """True when an embedded image likely represents a scanned page, not a logo/stamp."""

    width, height = _image_pixel_size(img)
    if width and height:
        long_edge = max(width, height)
        if long_edge >= _LARGE_RASTER_MIN_LONG_EDGE and (width * height) >= _LARGE_RASTER_MIN_AREA:
            return True

    try:
        data = img.data
        if data and len(data) >= _LARGE_RASTER_MIN_BYTES:
            return True
    except Exception:
        pass

    return False


def _page_has_large_raster_image(page: Any) -> bool:
    try:
        return any(_is_large_raster_image(img) for _, img in page.images.items())
    except Exception:
        return False


def _page_text_is_noisy_ocr(stripped: str) -> bool:
    """Heuristic for corrupted / low-quality OCR text on a single page."""

    chars = len(stripped)
    if chars < 40:
        return False

    alnum = sum(1 for c in stripped if c.isalnum() or c.isspace())
    common_syms = sum(1 for c in stripped if c in ".,-/():")
    alnum_plus_common = alnum + common_syms
    weird = sum(1 for c in stripped if c in "¦|\\_~[]{}<>^")
    replacement_chars = stripped.count("\ufffd")

    return (
        (alnum_plus_common / chars < 0.92)
        or (weird / chars > 0.02)
        or (replacement_chars > 0)
    )


def _extract_text_stats(pdf_path: Path) -> tuple[int, float, bool, bool, bool, bool]:
    """Return ``(page_count, text_density, has_text_layer, is_noisy_ocr, has_embedded_images, has_full_page_raster_image)``.

    ``text_density`` is a coarse metric: average characters per page divided by
    a reference size (2000 chars) and clamped to ``[0, 1]``. Any page emitting
    non-trivial text contributes to ``has_text_layer``.
    """

    try:
        from pypdf import PdfReader  # local import keeps startup cheap
    except Exception as exc:  # pragma: no cover - dependency failure path
        logger.warning("pypdf unavailable (%s); text layer detection disabled.", exc)
        return 0, 0.0, False, False, False, False

    try:
        reader = PdfReader(str(pdf_path))
        if getattr(reader, "is_encrypted", False):
            try:
                reader.decrypt("")
            except Exception as exc:
                logger.warning("pypdf could not decrypt %s: %s", pdf_path, exc)
                return 0, 0.0, False, False, False, False
    except Exception as exc:
        logger.warning("pypdf failed to open %s: %s", pdf_path, exc)
        return 0, 0.0, False, False, False, False

    try:
        page_count = len(reader.pages)
    except Exception as exc:
        logger.warning("pypdf failed to read page count for %s: %s", pdf_path, exc)
        return 0, 0.0, False, False, False, False
    if page_count == 0:
        return 0, 0.0, False, False, False, False

    non_trivial_pages = 0
    total_chars = 0
    noisy_pages = 0
    pages_with_embedded_images = 0
    pages_with_large_raster = 0

    for page in reader.pages:
        try:
            text = page.extract_text() or ""
            # pypdf images access can sometimes be slow or fail on corrupted PDFs
            has_embedded_images = len(page.images) > 0
            has_large_raster = _page_has_large_raster_image(page)
        except Exception:
            text = ""
            has_embedded_images = False
            has_large_raster = False

        stripped = text.strip()
        chars = len(stripped)
        total_chars += chars

        if chars >= 40:
            non_trivial_pages += 1
            if _page_text_is_noisy_ocr(stripped):
                noisy_pages += 1

        if has_embedded_images:
            pages_with_embedded_images += 1
        if has_large_raster:
            pages_with_large_raster += 1

    avg_chars = total_chars / page_count
    text_density = max(0.0, min(1.0, avg_chars / 2000.0))
    has_text_layer = non_trivial_pages >= max(1, int(0.5 * page_count))
    is_noisy_ocr = noisy_pages >= max(1, int(0.5 * page_count))
    has_embedded_images = pages_with_embedded_images >= max(1, int(0.5 * page_count))
    has_full_page_raster_image = pages_with_large_raster >= max(1, int(0.5 * page_count))

    return (
        page_count,
        text_density,
        has_text_layer,
        is_noisy_ocr,
        has_embedded_images,
        has_full_page_raster_image,
    )


def _sample_first_pages(pdf_path: Path, limit: int) -> list[np.ndarray]:
    poppler_kwargs: dict[str, Any] = {}
    if sys.platform == "win32" and getattr(settings, "poppler_path", None):
        poppler_kwargs["poppler_path"] = settings.poppler_path
    images: list[Image.Image] = convert_from_path(
        str(pdf_path),
        dpi=settings.pdf_dpi,
        fmt="png",
        grayscale=True,
        first_page=1,
        last_page=max(1, limit),
        **poppler_kwargs,
    )
    return [np.array(_smart_resize(img).convert("L")) for img in images]


def _table_presence_hint(gray: np.ndarray) -> bool:
    """Cheap hint: long nearly-horizontal dark runs are usually table rules."""

    if gray.size == 0:
        return False
    # Threshold to binary; treat "below mean - 30" as ink to avoid opencv import cost.
    threshold = max(0, int(gray.mean()) - 30)
    ink = gray < threshold
    # Longest run of ink per row
    max_run = 0
    for row in ink:
        if not row.any():
            continue
        # Find runs of True; quick numpy approach using diff on int
        row_int = row.astype(np.int8)
        diffs = np.diff(np.concatenate(([0], row_int, [0])))
        run_starts = np.where(diffs == 1)[0]
        run_ends = np.where(diffs == -1)[0]
        if run_starts.size == 0:
            continue
        longest = int((run_ends - run_starts).max())
        if longest > max_run:
            max_run = longest
    # Any horizontal ink run spanning >= 40% of the image width is "probably a table rule".
    return max_run >= int(0.4 * gray.shape[1])


def _classify(
    *,
    has_text_layer: bool,
    blur_score: float,
    noise_score: float,
    text_density: float,
    is_noisy_ocr: bool = False,
    has_full_page_raster_image: bool = False,
) -> tuple[QualityClass, list[str]]:
    reasons: list[str] = []

    is_blurry_hard = blur_score < BLUR_DEGRADED_THRESHOLD
    is_blurry_moderate = blur_score < BLUR_MODERATE_THRESHOLD
    is_noisy_hard = noise_score >= NOISE_DEGRADED_THRESHOLD
    is_noisy_moderate = noise_score >= NOISE_MODERATE_THRESHOLD

    # doc008 guard: corrupted OCR plus a page-sized raster is a noisy scan.
    if is_noisy_ocr and has_full_page_raster_image:
        reasons.append("ocr_text_layer_corrupted")
        reasons.append("full_page_raster_image")
        reasons.append("low_identifier_legibility")
        return "noisy_scan", reasons

    if has_text_layer and text_density >= DIGITAL_TEXT_DENSITY_MIN and not (is_blurry_hard or is_noisy_hard):
        if is_noisy_ocr:
            reasons.append("ocr_text_layer_corrupted")
            return "noisy_scan", reasons
        if has_full_page_raster_image:
            # Text layer over a full-page raster is almost always OCR on a scan.
            reasons.append("ocr_text_layer_corrupted")
            reasons.append("full_page_raster_image")
            reasons.append("low_identifier_legibility")
            return "noisy_scan", reasons

        reasons.append(f"text_layer_present(text_density={text_density:.2f})")
        return "digital_clean", reasons

    if is_blurry_hard and is_noisy_hard:
        if blur_score < BLUR_SEVERE_THRESHOLD or noise_score >= NOISE_SEVERE_THRESHOLD:
            reasons.append(f"blur_severe(blur_score={blur_score:.1f})")
            reasons.append(f"noise_severe(noise_score={noise_score:.1f})")
            return "severe_scan", reasons
        reasons.append(f"blur_high(blur_score={blur_score:.1f})")
        reasons.append(f"noise_high(noise_score={noise_score:.1f})")
        return "noisy_scan", reasons
    if is_blurry_hard:
        if blur_score < BLUR_SEVERE_THRESHOLD and text_density < DIGITAL_TEXT_DENSITY_MIN:
            reasons.append(f"blur_severe(blur_score={blur_score:.1f})")
            return "severe_scan", reasons
        reasons.append(f"blur_high(blur_score={blur_score:.1f})")
        return "noisy_scan", reasons
    if is_noisy_hard:
        if noise_score >= NOISE_SEVERE_THRESHOLD and text_density < DIGITAL_TEXT_DENSITY_MIN:
            reasons.append(f"noise_severe(noise_score={noise_score:.1f})")
            return "severe_scan", reasons
        reasons.append(f"noise_high(noise_score={noise_score:.1f})")
        return "noisy_scan", reasons
    if is_blurry_moderate and is_noisy_moderate:
        reasons.append(f"blur_moderate(blur_score={blur_score:.1f})")
        reasons.append(f"noise_moderate(noise_score={noise_score:.1f})")
        return "noisy_scan", reasons

    if has_text_layer and text_density >= DIGITAL_TEXT_DENSITY_MIN:
        # Text layer present but image quality is borderline; treat as noisy_scan
        # so the runtime still uses rasterised images with moderate preprocessing.
        reasons.append("text_layer_present_but_moderate_quality")
        return "noisy_scan", reasons

    reasons.append("no_text_layer")
    if is_blurry_moderate:
        reasons.append(f"blur_moderate(blur_score={blur_score:.1f})")
    if is_noisy_moderate:
        reasons.append(f"noise_moderate(noise_score={noise_score:.1f})")
    return "scan_clean", reasons


def classify_profile(
    *,
    has_text_layer: bool,
    blur_score: float,
    noise_score: float,
    text_density: float,
    is_noisy_ocr: bool = False,
    has_full_page_raster_image: bool = False,
) -> tuple[QualityClass, list[str]]:
    """Public helper so tests can exercise classification without PDF I/O."""

    return _classify(
        has_text_layer=has_text_layer,
        blur_score=blur_score,
        noise_score=noise_score,
        text_density=text_density,
        is_noisy_ocr=is_noisy_ocr,
        has_full_page_raster_image=has_full_page_raster_image,
    )


def profile_document(pdf_path: str | Path, *, document_id: str | None = None) -> DocumentProfile:
    """Profile a single PDF and return a :class:`DocumentProfile`.

    Rasterises only the first :data:`PROFILE_MAX_SAMPLE_PAGES` pages to estimate
    blur/noise. Text-layer detection runs on every page (it is cheap).
    """

    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    doc_id = document_id or path.stem

    (
        page_count,
        text_density,
        has_text_layer,
        is_noisy_ocr,
        has_embedded_images,
        has_full_page_raster_image,
    ) = _extract_text_stats(path)

    # If pypdf failed (page_count == 0) we still try to rasterise to keep going.
    blur_score = 0.0
    noise_score = 0.0
    table_presence = False

    try:
        sample_pages = _sample_first_pages(path, PROFILE_MAX_SAMPLE_PAGES)
    except Exception as exc:
        logger.warning("Rasterisation failed for %s: %s", path.name, exc)
        sample_pages = []

    if sample_pages:
        blur_values = [compute_blur_score(page) for page in sample_pages]
        noise_values = [estimate_noise(page) for page in sample_pages]
        blur_score = float(min(blur_values))  # worst-page blur
        noise_score = float(max(noise_values))  # worst-page noise
        table_presence = any(_table_presence_hint(page) for page in sample_pages)
        if page_count == 0:
            # pypdf could not open it, fall back to sample count as a lower bound.
            page_count = len(sample_pages)

    quality_class, reasons = _classify(
        has_text_layer=has_text_layer,
        blur_score=blur_score,
        noise_score=noise_score,
        text_density=text_density,
        is_noisy_ocr=is_noisy_ocr,
        has_full_page_raster_image=has_full_page_raster_image,
    )

    profile = DocumentProfile(
        document_id=doc_id,
        filename=path.name,
        page_count=page_count,
        has_text_layer=has_text_layer,
        text_density=round(text_density, 4),
        blur_score=round(blur_score, 2),
        noise_score=round(noise_score, 2),
        table_presence_hint=bool(table_presence),
        quality_class=quality_class,
        reasons=reasons,
        profile_confidence=round(
            0.95
            if quality_class == "digital_clean"
            else 0.82
            if quality_class == "scan_clean"
            else 0.68
            if quality_class == "noisy_scan"
            else 0.55,
            2,
        ),
        metrics_summary={
            "has_extractable_text": has_text_layer,
            "text_density": round(text_density, 4),
            "blur_score": round(blur_score, 2),
            "noise_score": round(noise_score, 2),
            "table_presence_hint": bool(table_presence),
            "is_noisy_ocr": bool(is_noisy_ocr),
            "has_embedded_images": bool(has_embedded_images),
            "has_full_page_raster_image": bool(has_full_page_raster_image),
            "has_images": bool(has_embedded_images),
            "ocr_likelihood": (
                "low"
                if quality_class == "digital_clean"
                else "medium"
                if quality_class == "scan_clean"
                else "high"
                if quality_class == "noisy_scan"
                else "very_high"
            ),
        },
    )
    logger.info(
        "Profiled %s: pages=%d quality=%s has_text=%s blur=%.1f noise=%.1f",
        path.name,
        page_count,
        quality_class,
        has_text_layer,
        blur_score,
        noise_score,
    )
    return profile
