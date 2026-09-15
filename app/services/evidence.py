from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from app.services.preprocessing import ProcessedPage
from config.settings import get_settings

EvidenceRole = Literal["full_page", "table_crop", "binary"]

_JSON_OVERHEAD_PER_IMAGE = 160
_DATA_URL_PREFIX_LEN = len("data:image/jpeg;base64,")


@dataclass(frozen=True)
class EvidenceBlock:
    page_number: int
    variant_name: str
    media_type: str
    data: str
    byte_size: int
    bbox: tuple[int, int, int, int] | None
    role: EvidenceRole


def encoded_request_bytes(byte_size: int) -> int:
    """Approximate JSON body contribution of a base64 data-URL image."""
    return (byte_size * 4 + 2) // 3 + _DATA_URL_PREFIX_LEN + _JSON_OVERHEAD_PER_IMAGE


def estimate_request_bytes(blocks: list[EvidenceBlock], prompt_overhead_bytes: int) -> int:
    return prompt_overhead_bytes + sum(encoded_request_bytes(block.byte_size) for block in blocks)


def _block_from_variant(
    page: ProcessedPage,
    variant_name: str,
    role: EvidenceRole,
) -> EvidenceBlock | None:
    image = page.variants.get(variant_name)
    if image is None:
        return None
    bbox = page.table_crop_bbox if role == "table_crop" else None
    return EvidenceBlock(
        page_number=page.page_number,
        variant_name=variant_name,
        media_type=image.media_type,
        data=image.data,
        byte_size=image.byte_size,
        bbox=bbox,
        role=role,
    )


def select_metadata_evidence(pages: list[ProcessedPage]) -> list[EvidenceBlock]:
    settings = get_settings()
    selected: list[EvidenceBlock] = []
    seen: set[tuple[int, str]] = set()
    for page in pages[: settings.max_pages_for_llm]:
        primary = page.full_page_variant_name if page.full_page_variant_name in page.variants else "contrast"
        if primary not in page.variants:
            primary = "full_gray"
        block = _block_from_variant(page, primary, "full_page")
        if block and (block.page_number, block.variant_name) not in seen:
            selected.append(block)
            seen.add((block.page_number, block.variant_name))
        if page.page_number == 1 and primary != "full_gray":
            extra = _block_from_variant(page, "full_gray", "full_page")
            if extra and (extra.page_number, extra.variant_name) not in seen:
                selected.append(extra)
                seen.add((extra.page_number, extra.variant_name))
    return selected


def select_row_evidence(pages: list[ProcessedPage]) -> list[EvidenceBlock]:
    settings = get_settings()
    selected: list[EvidenceBlock] = []
    seen: set[tuple[int, str]] = set()

    def _add(block: EvidenceBlock | None) -> None:
        if block is None or (block.page_number, block.variant_name) in seen:
            return
        selected.append(block)
        seen.add((block.page_number, block.variant_name))

    for page in pages[: settings.max_pages_for_llm]:
        if page.table_crop_available:
            _add(_block_from_variant(page, "table_crop", "table_crop"))
            _add(_block_from_variant(page, "adaptive_binary", "binary"))
        preferred = page.selected_variants if page.selected_variants else []
        for variant_name in preferred:
            role: EvidenceRole = "binary" if variant_name == "adaptive_binary" else "full_page"
            if variant_name == "table_crop":
                role = "table_crop"
            _add(_block_from_variant(page, variant_name, role))
        if not page.table_crop_available:
            full_page_name = page.full_page_variant_name if page.full_page_variant_name in page.variants else "contrast"
            _add(_block_from_variant(page, full_page_name, "full_page"))
    return selected


def pack_evidence(
    blocks: list[EvidenceBlock],
    *,
    max_request_bytes: int,
    prompt_overhead_bytes: int,
) -> list[EvidenceBlock]:
    """Keep preferred evidence while staying under the Mantle request-body cap."""
    if not blocks:
        return []
    ordered = sorted(
        blocks,
        key=lambda block: ({"table_crop": 0, "full_page": 1, "binary": 2}[block.role], block.page_number),
    )
    packed: list[EvidenceBlock] = []
    for block in ordered:
        candidate = packed + [block]
        if estimate_request_bytes(candidate, prompt_overhead_bytes) <= max_request_bytes:
            packed.append(block)
    if packed:
        return packed
    smallest = min(ordered, key=lambda block: block.byte_size)
    return [smallest]


def evidence_index(blocks: list[EvidenceBlock]) -> list[dict[str, Any]]:
    return [
        {
            "page_number": block.page_number,
            "variant_name": block.variant_name,
            "role": block.role,
            "bbox": list(block.bbox) if block.bbox is not None else None,
            "byte_size": block.byte_size,
        }
        for block in blocks
    ]


def to_openai_image_content(blocks: list[EvidenceBlock], trailing_text: str) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for block in blocks:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{block.media_type};base64,{block.data}"},
            }
        )
    if trailing_text:
        content.append({"type": "text", "text": trailing_text})
    return content


def to_anthropic_image_blocks(blocks: list[EvidenceBlock]) -> list[dict[str, Any]]:
    packed: list[dict[str, Any]] = []
    for block in blocks:
        packed.append(
            {
                "type": "image",
                "source": {"type": "base64", "media_type": block.media_type, "data": block.data},
            }
        )
    return packed
