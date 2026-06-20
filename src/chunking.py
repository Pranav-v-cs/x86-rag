import re
import uuid
from typing import Optional

from src.models import Chunk, InstructionEntry, SourceCategory, SubsectionType

INSTRUCTION_HEADING_RE = re.compile(
    r"^([A-Z0-9][A-Z0-9a-z/]*(?:—[\w\s/]+)?)$"
)
SUBSECTION_HEADINGS: dict[str, SubsectionType] = {
    "opcode": SubsectionType.OPCODE_ENCODING,
    "opcode table": SubsectionType.OPCODE_ENCODING,
    "opcode/encoding": SubsectionType.OPCODE_ENCODING,
    "encoding": SubsectionType.OPCODE_ENCODING,
    "instruction": SubsectionType.OPCODE_ENCODING,
    "description": SubsectionType.DESCRIPTION,
    "operation": SubsectionType.OPERATION,
    "flags affected": SubsectionType.FLAGS_AFFECTED,
    "flags": SubsectionType.FLAGS_AFFECTED,
    "exceptions": SubsectionType.EXCEPTIONS,
    "protected mode exceptions": SubsectionType.EXCEPTIONS,
    "real-address mode exceptions": SubsectionType.EXCEPTIONS,
    "virtual-8086 mode exceptions": SubsectionType.EXCEPTIONS,
    "compatibility mode exceptions": SubsectionType.EXCEPTIONS,
    "64-bit mode exceptions": SubsectionType.EXCEPTIONS,
    "cpuid feature flag": SubsectionType.CPUID_FEATURE,
}

MIN_CHUNK_SIZE = 800
MAX_CHUNK_SIZE = 1200
OVERLAP = 150


def _extract_mnemonics(heading: str) -> list[str]:
    heading = heading.strip().rstrip("—")
    parts = re.split(r"[/,]", heading)
    mnemonics = []
    for part in parts:
        part = part.strip()
        if part:
            mnem = part.split("—")[0].strip()
            if mnem and re.match(r"^[A-Z0-9][A-Z0-9a-z]*$", mnem):
                mnemonics.append(mnem)
    return mnemonics if mnemonics else [heading.strip()]


def _detect_subsection(line: str) -> Optional[SubsectionType]:
    stripped = line.strip().rstrip(":").strip().lower()
    for key, stype in SUBSECTION_HEADINGS.items():
        if stripped == key or stripped.startswith(key):
            return stype
    return None


def _split_long_text(text: str, mnemonic: str, subsection: SubsectionType) -> list[str]:
    lines = text.split("\n")
    chunks: list[str] = []
    current_chunk: list[str] = []
    current_len = 0

    for line in lines:
        line_len = len(line) + 1
        if current_len + line_len > MAX_CHUNK_SIZE and current_len >= MIN_CHUNK_SIZE:
            chunks.append("\n".join(current_chunk))
            overlap_lines: list[str] = []
            overlap_len = 0
            for ol in reversed(current_chunk):
                if overlap_len + len(ol) + 1 > OVERLAP:
                    break
                overlap_lines.insert(0, ol)
                overlap_len += len(ol) + 1
            current_chunk = overlap_lines
            current_len = overlap_len

        current_chunk.append(line)
        current_len += line_len

    if current_chunk:
        chunks.append("\n".join(current_chunk))

    return chunks


def chunk_instruction_entry(
    entry: InstructionEntry, category: SourceCategory
) -> list[Chunk]:
    chunks: list[Chunk] = []
    mnemonic_str = "/".join(entry.mnemonics)

    for subsection, text in entry.subsections.items():
        full_text = f"{mnemonic_str} — {subsection.value}: {text}"
        text_len = len(full_text)

        if text_len <= MAX_CHUNK_SIZE:
            chunk_id = str(uuid.uuid4())
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    mnemonic=mnemonic_str,
                    subsection=subsection,
                    text=full_text,
                    source=entry.source,
                    page=entry.page,
                    category=category,
                )
            )
        else:
            sub_chunks = _split_long_text(full_text, mnemonic_str, subsection)
            for sc in sub_chunks:
                chunk_id = str(uuid.uuid4())
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        mnemonic=mnemonic_str,
                        subsection=subsection,
                        text=sc,
                        source=entry.source,
                        page=entry.page,
                        category=category,
                    )
                )

    return chunks


def parse_instruction_entries(
    text: str, source: str, page: Optional[int], category: SourceCategory
) -> list[InstructionEntry]:
    entries: list[InstructionEntry] = []
    lines = text.split("\n")
    current_entry: Optional[dict] = None
    current_subsection: Optional[SubsectionType] = None
    current_sub_lines: list[str] = []

    def _flush_subsection():
        if current_entry is not None and current_subsection is not None and current_sub_lines:
            text_blob = "\n".join(current_sub_lines).strip()
            if text_blob:
                current_entry["subsections"][current_subsection] = text_blob

    def _flush_entry():
        nonlocal current_entry, current_subsection, current_sub_lines
        _flush_subsection()
        if current_entry is not None:
            entries.append(
                InstructionEntry(
                    mnemonics=current_entry["mnemonics"],
                    subsections=current_entry["subsections"],
                    source=current_entry["source"],
                    page=current_entry["page"],
                    category=current_entry["category"],
                )
            )
        current_entry = None
        current_subsection = None
        current_sub_lines = []

    for i, line in enumerate(lines):
        stripped = line.strip()

        if not stripped:
            continue

        heading_match = INSTRUCTION_HEADING_RE.match(stripped)
        if heading_match:
            _flush_entry()
            mnemonics = _extract_mnemonics(stripped)
            current_entry = {
                "mnemonics": mnemonics,
                "subsections": {},
                "source": source,
                "page": page,
                "category": category,
            }
            current_subsection = None
            current_sub_lines = []
            continue

        subsection_type = _detect_subsection(line)
        if subsection_type is not None:
            _flush_subsection()
            current_subsection = subsection_type
            current_sub_lines = []
            text_after_colon = re.sub(r"^[^:]*:\s*", "", line).strip()
            if text_after_colon:
                current_sub_lines.append(text_after_colon)
            continue

        if current_subsection is not None:
            current_sub_lines.append(line)

    _flush_entry()

    return entries
