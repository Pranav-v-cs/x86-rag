from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class SourceCategory(str, Enum):
    SDM_VOL1 = "sdm_vol1"
    SDM_VOL2 = "sdm_vol2"
    HTML_REF = "html_ref"


class SubsectionType(str, Enum):
    OPCODE_ENCODING = "opcode_encoding"
    DESCRIPTION = "description"
    OPERATION = "operation"
    FLAGS_AFFECTED = "flags_affected"
    EXCEPTIONS = "exceptions"
    CPUID_FEATURE = "cpuid_feature"
    OTHER = "other"


@dataclass
class Chunk:
    chunk_id: str
    mnemonic: str
    subsection: SubsectionType
    text: str
    source: str
    page: Optional[int]
    category: SourceCategory
    metadata: dict = field(default_factory=dict)


@dataclass
class InstructionEntry:
    mnemonics: list[str]
    subsections: dict[SubsectionType, str]
    source: str
    page: Optional[int]
    category: SourceCategory


@dataclass
class RetrievalResult:
    chunks: list[Chunk]
    query: str
    bm25_score: float = 0.0
    vector_score: float = 0.0
    fused_score: float = 0.0


@dataclass
class SourceDocument:
    filename: str
    category: SourceCategory
    num_chunks: int = 0
    num_instructions: int = 0


@dataclass
class Stats:
    num_documents: int
    num_chunks: int
    num_unique_mnemonics: int
    vector_db_size_bytes: int
    num_files_by_category: dict[str, int]
