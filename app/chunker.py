"""Split a document into retrieval-sized chunks.

Two levels:

1. Split the doc on markdown headers first, so each section stays topically
   self-contained (a "## Mutex vs Semaphore" section doesn't get split across
   two unrelated chunks).
2. If a section is still bigger than chunk_size, fall back to recursive
   character splitting: try to cut on paragraph breaks, then lines, then
   sentences, then words, only dropping to a finer separator for the pieces
   that are still too big. This works on any text, headers or not, unlike a
   splitter that only understands paragraphs. Consecutive chunks produced
   this way share `overlap` characters of trailing text so context isn't
   lost at the cut.

Overlap is only applied within an oversized section. Separate header sections
are natural boundaries and are not overlapped.
"""
import re

from app import config

_HEADER_RE = re.compile(r"^(#{1,6})\s+(.*)")

# Tried in order, most semantic first; "" (single characters) is the
# guaranteed-to-terminate last resort.
_SEPARATORS = ("\n\n", "\n", ". ", " ", "")


# Split markdown text into (header_text, section_text) tuples. The header line
# is kept as the first line of the section that follows it. Text before the
# first header (or a doc with no headers) comes back with header_text="".
# Empty sections are dropped.
def _split_by_headers(text):
    sections = []
    current_header = ""
    current_lines = []

    for line in text.split("\n"):
        match = _HEADER_RE.match(line)
        if match:
            if current_lines:  # flush the section built up before this header
                sections.append((current_header, "\n".join(current_lines).strip()))
            current_header = match.group(2).strip()
            current_lines = [line]
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((current_header, "\n".join(current_lines).strip()))

    return [s for s in sections if s[1]]


# Last `overlap` characters of `text`, snapped forward to the next word boundary
# so the carried-over fragment doesn't start mid-word. Returns "" when overlap
# is 0 or there's nothing to carry.
def _overlap_tail(text, overlap):
    if overlap <= 0 or len(text) <= overlap:
        return text if overlap > 0 else ""
    tail = text[-overlap:]
    space = tail.find(" ")
    return tail[space + 1:] if space != -1 else tail


# Recursively split `text` on the separators in `seps`, in order. Parts that
# already fit chunk_size are batched up and packed together (via
# _merge_pieces) using the current separator; a part that's still too big
# flushes whatever's batched so far, then gets split further with the next
# (finer) separator -- its result is emitted as-is, not re-packed with
# neighboring parts, so an already chunk_size-sized piece never gets glued to
# another one and pushed back over budget.
def _recursive_split(text, chunk_size, overlap, seps=_SEPARATORS):
    sep, rest = seps[0], seps[1:]
    parts = list(text) if sep == "" else [p for p in text.split(sep) if p]

    chunks = []
    batch = []
    for part in parts:
        if len(part) <= chunk_size:
            batch.append(part)
            continue
        if batch:
            chunks.extend(_merge_pieces(batch, sep, chunk_size, overlap))
            batch = []
        chunks.extend(_recursive_split(part, chunk_size, overlap, rest) if rest else [part])

    if batch:
        chunks.extend(_merge_pieces(batch, sep, chunk_size, overlap))

    return chunks


# Greedily pack `pieces` back together (joined by `separator`) up to
# chunk_size. Each new chunk is seeded with the overlap tail of the previous
# one; the seed alone is never emitted as its own chunk (it only appears once
# real content has been added on top of it), so overlap never becomes a
# chunk by itself.
def _merge_pieces(pieces, separator, chunk_size, overlap):
    chunks = []
    current = ""
    seeded_len = 0

    for piece in pieces:
        candidate = f"{current}{separator}{piece}" if current else piece
        if len(candidate) > chunk_size and current and len(current) > seeded_len:
            chunks.append(current)
            current = _overlap_tail(current, overlap)
            seeded_len = len(current)
            candidate = f"{current}{separator}{piece}" if current else piece
        current = candidate

    if current and len(current) > seeded_len:
        chunks.append(current)

    return chunks


# Split a single (possibly header-prefixed) section that's over chunk_size.
# `header`, if any, is re-attached as a "## <header>" prefix to every
# resulting chunk so the section's topic travels with each piece, instead of
# ending up alone in an orphan chunk. The size budget is reduced by the
# prefix length so real chunk_size is still respected.
def _split_section(section_text, chunk_size, overlap, header=""):
    body = section_text
    if header:
        # Drop the header line itself so it isn't treated as ordinary text.
        body = re.sub(rf"^#{{1,6}}\s+{re.escape(header)}\s*\n?", "", section_text, count=1).strip()

    prefix = f"## {header}\n\n" if header else ""
    effective_size = max(chunk_size - len(prefix), 200)
    # Never carry more than half the budget, or chunks barely advance.
    overlap = max(0, min(overlap, effective_size // 2))

    chunks = [
        (prefix + piece).strip()
        for piece in _recursive_split(body, effective_size, overlap)
        if piece.strip()
    ]
    return chunks if chunks else ([prefix.strip()] if prefix else [])


# Public entry point. Returns a flat list of chunk strings ready to embed.
def chunk_text(text, chunk_size=None, overlap=None):
    chunk_size = chunk_size or config.CHUNK_SIZE_CHARS
    overlap = config.CHUNK_OVERLAP_CHARS if overlap is None else overlap

    text = text.strip()
    if not text:
        return []

    all_chunks = []
    for header, section_text in _split_by_headers(text):
        if len(section_text) <= chunk_size:
            all_chunks.append(section_text)
        else:
            all_chunks.extend(_split_section(section_text, chunk_size, overlap, header))

    return all_chunks
