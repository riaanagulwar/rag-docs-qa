"""Tests for app.chunker: header-aware splitting (markdown and heuristic
plain-text headers), the recursive-splitting fallback, and overlap."""
from pathlib import Path

from app.chunker import chunk_text

DOCS_DIR = Path(__file__).parent.parent / "docs"


def test_markdown_headers_split_into_one_chunk_per_section():
    doc = """# Title

Intro paragraph here.

## Section A

Some content for section A.

## Section B

Some content for section B.
"""
    chunks = chunk_text(doc, chunk_size=1800)
    assert len(chunks) == 3
    assert chunks[1].startswith("## Section A")
    assert chunks[2].startswith("## Section B")


def test_heuristic_headers_split_plain_text_the_same_way():
    # Same structure as the markdown test, but no "#" anywhere -- isolated,
    # title-cased lines should be detected as headers too.
    doc = """Title

Intro paragraph here.

Section A

Some content for section A.

Section B

Some content for section B.
"""
    chunks = chunk_text(doc, chunk_size=1800)
    assert len(chunks) == 3
    assert chunks[1].startswith("Section A")
    assert chunks[2].startswith("Section B")


def test_heuristic_header_does_not_misfire_on_a_normal_sentence():
    # A short, isolated, unpunctuated line that reads like a sentence (not a
    # title) must NOT be treated as a heading -- regression test for the
    # false positive caught during development ("Rome was not built in a day").
    doc = """This is the first paragraph of a long document with no structure at all.
It just keeps going for a while to build up some length for the test.

Rome was not built in a day

But it eventually became a very large empire that lasted for centuries and
had significant influence on law, architecture, and language across Europe.
"""
    chunks = chunk_text(doc, chunk_size=200, overlap=30)
    assert not any(c.startswith("Rome was not built in a day") for c in chunks)


def test_oversized_section_falls_back_to_recursive_split_with_overlap():
    paragraphs = "\n\n".join(
        f"Paragraph number {i} has some filler text to take up space here." for i in range(40)
    )
    doc = f"""# Title

## Section A

{paragraphs}
"""
    chunks = chunk_text(doc, chunk_size=500, overlap=120)
    section_a_chunks = [c for c in chunks if c.startswith("## Section A")]
    assert len(section_a_chunks) > 1
    assert all(len(c) <= 700 for c in chunks)  # bounded, allowing some overlap slack

    # consecutive chunks should share trailing/leading text (the overlap)
    for a, b in zip(section_a_chunks, section_a_chunks[1:]):
        tail_word = a[-60:].split()[-3]
        assert tail_word in b


def test_pathological_input_with_no_separators_still_terminates_and_bounds_size():
    chunks = chunk_text("x" * 5000, chunk_size=300, overlap=50)
    assert len(chunks) > 1
    assert all(len(c) <= 350 for c in chunks)


def test_empty_and_whitespace_input_returns_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   \n\n  ") == []


def test_sample_docs_each_chunk_one_section_per_header():
    for name in ["test-doc1.md", "test-doc2.md", "test-doc3.md"]:
        text = (DOCS_DIR / name).read_text()
        chunks = chunk_text(text)
        # title line + 5 sections, per the fixed structure of these docs
        assert len(chunks) == 6, f"{name}: expected 6 chunks, got {len(chunks)}"
