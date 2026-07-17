import re

from app import config


def _split_by_headers(text: str):
    """
    Splits markdown text on headers (#, ##, ###...), keeping the header
    attached to the section that follows it. Returns a list of
    (header_path, section_text) tuples. If there are no headers at all,
    returns a single section with header_path="".
    """
    lines = text.split("\n")
    sections = []
    current_header = ""
    current_lines = []

    header_pattern = re.compile(r"^(#{1,6})\s+(.*)")

    for line in lines:
        match = header_pattern.match(line)
        if match:
            # flush whatever we were building before this header
            if current_lines:
                sections.append((current_header, "\n".join(current_lines).strip()))
            current_header = match.group(2).strip()
            current_lines = [line]  # keep header text as part of the section
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((current_header, "\n".join(current_lines).strip()))

    return [s for s in sections if s[1]]


def _split_on_paragraph_boundary(text: str, chunk_size: int, overlap: int, header: str = ""):
    """
    Fallback splitter for sections still too large after header splitting,
    or for docs with no headers at all. Grows a chunk paragraph by
    paragraph (falling back to sentence, then hard cut) until it hits
    chunk_size, instead of cutting at an arbitrary character index.

    The header (if any) is treated as a prefix re-attached to every
    resulting chunk, rather than as content that can end up alone in
    its own orphan chunk.
    """
    # Strip the header line out of the body so it's never treated as
    # just another paragraph that can get flushed on its own.
    body = text
    if header:
        body = re.sub(rf"^#{{1,6}}\s+{re.escape(header)}\s*\n?", "", text, count=1).strip()

    prefix = f"## {header}\n\n" if header else ""
    # reserve room for the header prefix so real chunk_size is respected
    effective_size = max(chunk_size - len(prefix), 200)

    paragraphs = re.split(r"\n\s*\n", body)
    chunks = []
    current = ""

    def flush():
        if current.strip():
            chunks.append((prefix + current).strip())

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # A single paragraph longer than chunk_size: split on sentences
        if len(para) > effective_size:
            sentences = re.split(r"(?<=[.!?])\s+", para)
            for sent in sentences:
                if current and len(current) + len(sent) + 1 > effective_size:
                    flush()
                    current = sent
                else:
                    current = f"{current} {sent}".strip()
            continue

        if current and len(current) + len(para) + 2 > effective_size:
            flush()
            current = para
        else:
            current = f"{current}\n\n{para}".strip()

    flush()

    return chunks if chunks else ([prefix.strip()] if prefix else [])


def chunk_text(text: str, chunk_size: int = None, overlap: int = None):
    """
    Header-aware chunking with paragraph/sentence-boundary fallback.

    Strategy:
    1. Split the doc on markdown headers first, so each section stays
       topically self-contained (a "## Mutex vs Semaphore" section
       doesn't get split across two unrelated chunks).
    2. If a section is still bigger than chunk_size, break it further
       on paragraph boundaries (falling back to sentence boundaries),
       instead of cutting at a raw character count.
    """
    chunk_size = chunk_size or config.CHUNK_SIZE_CHARS
    overlap = overlap or config.CHUNK_OVERLAP_CHARS

    text = text.strip()
    if not text:
        return []

    sections = _split_by_headers(text)

    all_chunks = []
    for header, section_text in sections:
        if len(section_text) <= chunk_size:
            all_chunks.append(section_text)
        else:
            all_chunks.extend(
                _split_on_paragraph_boundary(section_text, chunk_size, overlap, header)
            )

    return all_chunks
