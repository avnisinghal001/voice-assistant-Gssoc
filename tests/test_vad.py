from __future__ import annotations

import struct

from voice_assistant.asr.vad import VADConfig, VoiceActivityDetector


def _pcm_frame(value: int, samples: int) -> bytes:
    return struct.pack("<" + "h" * samples, *([value] * samples))


def test_vad_energy_mode_detects_speech_after_consecutive_frames() -> None:
    cfg = VADConfig(sample_rate=16_000, frame_ms=30, speech_frames_trigger=3, mode="energy")
    vad = VoiceActivityDetector(cfg)
    samples = int(cfg.sample_rate * cfg.frame_ms / 1000)

    silence = _pcm_frame(0, samples)
    speech = _pcm_frame(2000, samples)

    assert vad.is_speech(silence) is False
    assert vad.detect_barge_in([speech, speech]) is False
    assert vad.detect_barge_in([speech, speech, speech]) is True
