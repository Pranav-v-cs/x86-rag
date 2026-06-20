import logging
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional

from bs4 import BeautifulSoup
import fitz

from src.chunking import chunk_instruction_entry, parse_instruction_entries
from src.models import Chunk, InstructionEntry, SourceCategory

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

INSTRUCTION_MNEMONIC_RE = re.compile(r"^([A-Z0-9][A-Z0-9a-z]*(?:/[A-Z0-9][A-Z0-9a-z]*)*)—")


def _cluster_text_blocks(page) -> str:
    blocks = page.get_text("blocks")
    if not blocks:
        return page.get_text()

    x_coords = [(b[0], b[1], b[2], b[4]) for b in blocks]
    x_starts = sorted(set(round(b[0], -1) for b in x_coords))

    if len(x_starts) <= 1:
        return page.get_text()

    tolerance = 20
    clusters = defaultdict(list)
    for b in blocks:
        x0 = round(b[0] / 10) * 10
        for xs in x_starts:
            if abs(x0 - xs) <= tolerance:
                clusters[xs].append(b)
                break
        else:
            clusters[x0].append(b)

    sorted_clusters = sorted(clusters.items(), key=lambda kv: kv[0])
    lines: list[tuple[float, str]] = []
    for x_start, cluster_blocks in sorted_clusters:
        for b in cluster_blocks:
            y0 = b[1]
            text = b[4].strip()
            if text:
                lines.append((y0, text))
        lines.append((float("inf"), ""))

    lines.sort(key=lambda x: x[0])
    return "\n".join(text for _, text in lines)


def _extract_text_from_pdf(filepath: str) -> dict[int, str]:
    doc = fitz.open(filepath)
    page_texts: dict[int, str] = {}
    for i, page in enumerate(doc):
        try:
            text = _cluster_text_blocks(page)
            if text.strip():
                page_texts[i + 1] = text
        except Exception as e:
            logger.warning("Failed to extract page %d from %s: %s", i + 1, filepath, e)
    doc.close()
    return page_texts


def _extract_entries_from_pdf_page(
    text: str, filename: str, page_num: int, category: SourceCategory
) -> list[InstructionEntry]:
    return parse_instruction_entries(text, filename, page_num, category)


def _extract_entries_from_html(filepath: str) -> list[InstructionEntry]:
    entries: list[InstructionEntry] = []
    filename = os.path.basename(filepath)

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        soup = BeautifulSoup(f, "lxml")

    for tag in soup.find_all(["h1", "h2", "h3", "h4"]):
        heading_text = tag.get_text(strip=True)
        mnemonic_match = INSTRUCTION_MNEMONIC_RE.match(heading_text)
        if not mnemonic_match:
            continue

        mnemonics = [m.strip() for m in mnemonic_match.group(1).split("/")]
        subsections: dict = {}

        current_subsection = "description"
        current_lines: list[str] = []

        def _flush():
            if current_lines:
                text = "\n".join(current_lines).strip()
                if text:
                    subsections[current_subsection] = text

        sibling = tag.find_next_sibling()
        while sibling and sibling.name not in ("h1", "h2", "h3", "h4"):
            text = sibling.get_text(strip=True)
            if sibling.name in ("table", "pre", "code"):
                if current_lines:
                    current_lines.append("")
                current_lines.append(sibling.get_text("\n", strip=True))
                current_lines.append("")
            else:
                lower_text = text.lower().rstrip(":").strip()
                if lower_text in (
                    "opcode", "opcode table", "opcode/encoding",
                    "description", "operation", "flags affected",
                    "exceptions", "cpuid feature flag",
                ):
                    _flush()
                    current_subsection = lower_text
                    current_lines = []
                elif text:
                    current_lines.append(text)
            sibling = sibling.find_next_sibling()

        _flush()

        if subsections:
            entries.append(
                InstructionEntry(
                    mnemonics=mnemonics,
                    subsections=subsections,
                    source=filename,
                    page=None,
                    category=SourceCategory.HTML_REF,
                )
            )

    return entries


def ingest_pdf_directory(dirpath: str, category: SourceCategory) -> list[Chunk]:
    chunks: list[Chunk] = []
    path = Path(dirpath)
    if not path.exists():
        logger.info("Directory %s does not exist, skipping.", dirpath)
        return chunks

    pdf_files = sorted(path.glob("*.pdf"))
    if not pdf_files:
        logger.info("No PDF files found in %s.", dirpath)
        return chunks

    for pdf_path in pdf_files:
        filename = pdf_path.name
        logger.info("Processing PDF: %s", filename)
        try:
            page_texts = _extract_text_from_pdf(str(pdf_path))
            for page_num, text in page_texts.items():
                entries = _extract_entries_from_pdf_page(text, filename, page_num, category)
                for entry in entries:
                    try:
                        chunks.extend(chunk_instruction_entry(entry, category))
                    except Exception as e:
                        logger.warning("Failed to chunk entry %s: %s", entry.mnemonics, e)
        except Exception as e:
            logger.warning("Failed to process PDF %s: %s", filename, e)

    logger.info("Extracted %d chunks from %s.", len(chunks), dirpath)
    return chunks


def ingest_html_directory(dirpath: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    path = Path(dirpath)
    if not path.exists():
        logger.info("Directory %s does not exist, skipping.", dirpath)
        return chunks

    html_files = sorted(path.glob("*.html"))
    if not html_files:
        html_files = sorted(path.glob("*.htm"))
    if not html_files:
        logger.info("No HTML files found in %s.", dirpath)
        return chunks

    for html_path in html_files:
        filename = html_path.name
        logger.info("Processing HTML: %s", filename)
        try:
            entries = _extract_entries_from_html(str(html_path))
            for entry in entries:
                try:
                    chunks.extend(chunk_instruction_entry(entry, SourceCategory.HTML_REF))
                except Exception as e:
                    logger.warning("Failed to chunk HTML entry %s: %s", entry.mnemonics, e)
        except Exception as e:
            logger.warning("Failed to process HTML %s: %s", filename, e)

    logger.info("Extracted %d chunks from %s.", len(chunks), dirpath)
    return chunks


def ingest_all(
    sdm_vol1_dir: str = "docs/sdm_vol1",
    sdm_vol2_dir: str = "docs/sdm_vol2",
    html_ref_dir: str = "docs/html_ref",
) -> list[Chunk]:
    all_chunks: list[Chunk] = []

    all_chunks.extend(ingest_pdf_directory(sdm_vol1_dir, SourceCategory.SDM_VOL1))
    all_chunks.extend(ingest_pdf_directory(sdm_vol2_dir, SourceCategory.SDM_VOL2))
    all_chunks.extend(ingest_html_directory(html_ref_dir))

    logger.info("Total chunks extracted: %d", len(all_chunks))
    return all_chunks
