from __future__ import annotations

from app.services.evidence import (
    EvidenceBlock,
    estimate_request_bytes,
    pack_evidence,
    select_metadata_evidence,
)
from app.services.preprocessing import EncodedVariant, ProcessedPage


def _variant(size: int, data: str = "abc") -> EncodedVariant:
    return EncodedVariant(data=data, media_type="image/jpeg", width=100, height=100, byte_size=size)


def _page(*, page_number: int = 1, variants: dict[str, EncodedVariant] | None = None) -> ProcessedPage:
    variants = variants or {
        "contrast": _variant(200_000),
        "full_gray": _variant(180_000),
        "table_crop": _variant(90_000),
        "adaptive_binary": _variant(120_000),
    }
    return ProcessedPage(
        page_number=page_number,
        variants=variants,
        table_crop_bbox=(1, 2, 3, 4),
        table_crop_available=True,
        full_page_variant_name="contrast",
        selected_variants=["contrast", "table_crop"],
        primary_variant="contrast",
        detected_condition="digital_clean",
    )


def test_pack_evidence_stays_under_budget():
    blocks = [
        EvidenceBlock(1, "table_crop", "image/jpeg", "aa", 2_000_000, (0, 0, 1, 1), "table_crop"),
        EvidenceBlock(1, "full", "image/jpeg", "bb", 2_000_000, None, "full_page"),
        EvidenceBlock(1, "bin", "image/jpeg", "cc", 2_000_000, None, "binary"),
    ]
    packed = pack_evidence(blocks, max_request_bytes=3_500_000, prompt_overhead_bytes=80_000)
    assert packed
    assert estimate_request_bytes(packed, 80_000) <= 3_500_000
    assert packed[0].role == "table_crop"


def test_pack_evidence_keeps_full_page_instead_of_binary():
    blocks = [
        EvidenceBlock(1, "table_crop", "image/jpeg", "aa", 800_000, (0, 0, 1, 1), "table_crop"),
        EvidenceBlock(1, "full", "image/jpeg", "bb", 800_000, None, "full_page"),
        EvidenceBlock(1, "bin", "image/jpeg", "cc", 800_000, None, "binary"),
    ]
    packed = pack_evidence(blocks, max_request_bytes=3_000_000, prompt_overhead_bytes=80_000)
    roles = [block.role for block in packed]
    assert "table_crop" in roles
    assert "full_page" in roles
    assert "binary" not in roles
    assert estimate_request_bytes(packed, 80_000) <= 3_000_000


def test_metadata_evidence_includes_page_numbers():
    blocks = select_metadata_evidence([_page()])
    assert blocks
    assert all(block.page_number == 1 for block in blocks)


def test_sync_extract_rejects_non_pdf_magic(client, auth_headers):
    response = client.post(
        "/api/v1/extract",
        headers=auth_headers,
        files={"file": ("doc.pdf", b"not-a-pdf", "application/pdf")},
    )
    assert response.status_code == 415
