from __future__ import annotations

from voice_assistant.tts.stream import sentence_chunks_from_tokens


def test_sentence_chunker_splits_on_punctuation() -> None:
    tokens = ["Hello", " there.", " How", " are", " you?", " I", " am", " fine!"]
    chunks = sentence_chunks_from_tokens(tokens, max_tokens=50)

    assert chunks == ["Hello there.", "How are you?", "I am fine!"]


def test_sentence_chunker_limits_token_count() -> None:
    tokens = ["one ", "two ", "three ", "four ", "five ", "six ", "seven ", "eight "]
    chunks = sentence_chunks_from_tokens(tokens, max_tokens=3)
    assert len(chunks) >= 2
