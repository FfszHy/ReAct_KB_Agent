from __future__ import annotations

from itertools import pairwise

from pkb_agent.rag.chunking import (
    Chunk,
    count_tokens,
    split_into_chunks,
    split_text,
)

# ---- count_tokens ----------------------------------------------------------


def test_count_tokens_empty():
    assert count_tokens("") == 0


def test_count_tokens_nonempty_positive():
    assert count_tokens("hello world") > 0


def test_count_tokens_none_returns_zero():
    assert count_tokens(None) == 0  # type: ignore[arg-type]


# ---- split_text: edge cases -----------------------------------------------


def test_split_text_empty():
    assert split_text("") == []


def test_split_text_whitespace_only():
    assert split_text("   \n\t  ") == []


def test_split_text_zero_chunk_size():
    assert split_text("some text", chunk_size=0) == []


def test_split_text_negative_chunk_size():
    assert split_text("some text", chunk_size=-5) == []


# ---- split_text: basic splitting ------------------------------------------


def test_split_text_single_chunk_small_input():
    text = "A short sentence."
    chunks = split_text(text, chunk_size=800, chunk_overlap=0)
    assert len(chunks) == 1
    assert chunks[0].content.strip() == text.strip()
    assert chunks[0].chunk_index == 0


def test_split_text_chunk_size_respected():
    text = ". ".join(f"sentence number {i}" for i in range(50))
    chunks = split_text(text, chunk_size=20, chunk_overlap=0)
    assert len(chunks) >= 2
    for c in chunks:
        assert c.token_count <= 20


def test_split_text_chunk_index_sequential():
    text = ". ".join(f"marker {i} content" for i in range(40))
    chunks = split_text(text, chunk_size=15, chunk_overlap=0)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_split_text_char_spans_within_text():
    text = "first sentence. second sentence. third sentence."
    chunks = split_text(text, chunk_size=4, chunk_overlap=0)
    assert chunks
    for c in chunks:
        assert 0 <= c.char_start < c.char_end <= len(text)
        assert text[c.char_start:c.char_end] == c.content


# ---- overlap ---------------------------------------------------------------


def test_split_text_overlap_zero_disjoint():
    text = ". ".join(f"part {i}" for i in range(30))
    chunks = split_text(text, chunk_size=10, chunk_overlap=0)
    assert len(chunks) >= 2
    for prev, cur in pairwise(chunks):
        # With no overlap, consecutive chunks touch but do not share content.
        assert prev.char_end <= cur.char_start


def test_split_text_overlap_carried_between_chunks():
    text = ". ".join(f"item number {i}" for i in range(40))
    chunks = split_text(text, chunk_size=20, chunk_overlap=10)
    assert len(chunks) >= 2
    for prev, cur in pairwise(chunks):
        # With overlap, the next chunk starts before the previous ends.
        assert cur.char_start < prev.char_end


def test_split_text_overlap_capped_to_chunk_size_minus_one():
    text = ". ".join(f"word {i}" for i in range(20))
    # overlap larger than chunk_size should be capped; should not crash.
    chunks = split_text(text, chunk_size=20, chunk_overlap=1000)
    assert chunks


# ---- natural boundaries ---------------------------------------------------


def test_split_text_paragraph_boundaries():
    text = (
        "Para one content here.\n\n"
        "Para two content here.\n\n"
        "Para three content."
    )
    chunks = split_text(text, chunk_size=5, chunk_overlap=0)
    assert len(chunks) >= 2
    joined = "".join(c.content for c in chunks)
    assert "Para one" in joined
    assert "Para two" in joined
    assert "Para three" in joined


def test_split_text_cjk_boundaries():
    sentences = [f"第{i}句话内容。" for i in range(12)]
    text = "\n".join(sentences)
    chunks = split_text(text, chunk_size=2, chunk_overlap=0)
    assert len(chunks) >= 2
    joined = "".join(c.content for c in chunks)
    for i in range(12):
        assert f"第{i}句" in joined


# ---- split_into_chunks: base_index ----------------------------------------


def test_split_into_chunks_base_index_offset():
    text = ". ".join(f"seg {i}" for i in range(30))
    base = 5
    chunks = split_into_chunks(text, chunk_size=10, chunk_overlap=0, base_index=base)
    assert chunks
    assert [c.chunk_index for c in chunks] == [base + i for i in range(len(chunks))]


def test_split_into_chunks_base_index_zero_matches_split_text():
    text = "a sentence. b sentence. c sentence."
    a = split_text(text, chunk_size=10, chunk_overlap=0)
    b = split_into_chunks(text, chunk_size=10, chunk_overlap=0, base_index=0)
    assert len(a) == len(b)
    assert [c.chunk_index for c in b] == [c.chunk_index for c in a]


def test_chunk_dataclass_fields():
    c = Chunk(content="x", chunk_index=2, token_count=1, char_start=0, char_end=1)
    assert c.content == "x"
    assert c.chunk_index == 2
    assert c.token_count == 1
    assert c.char_start == 0
    assert c.char_end == 1
