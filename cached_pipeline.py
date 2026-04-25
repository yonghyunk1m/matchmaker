"""Cached pipeline helpers — speed up repeated experiments by storing
expensive intermediates (score audio render, CQT chroma) on disk.

Usage:
  from cached_pipeline import get_cached_score_features, get_cached_perf_features

The cache uses content+param hashing; same XML+tempo → same key. Cache lives
under cache/ in the repo root (gitignored).
"""
import hashlib
import os
import pickle
import sys
from pathlib import Path

import librosa
import numpy as np
import partitura as pt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CACHE_DIR = Path(os.path.dirname(os.path.abspath(__file__))) / "cache" / "pipeline"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_RATE = 44100


def _hash_inputs(*items):
    h = hashlib.md5()
    for it in items:
        if isinstance(it, (str, int, float)):
            h.update(str(it).encode())
        elif isinstance(it, bytes):
            h.update(it)
        else:
            h.update(repr(it).encode())
    return h.hexdigest()[:16]


def _file_mtime_size(path):
    s = os.stat(path)
    return (path, s.st_mtime, s.st_size)


def get_cached_score_audio(xml_path, tempo, sr=SAMPLE_RATE):
    """FluidSynth render of score with given tempo. Cached on disk."""
    from matchmaker.utils.misc import generate_score_audio
    key = _hash_inputs(*_file_mtime_size(xml_path), tempo, sr)
    cache_file = CACHE_DIR / f"score_audio_{key}.pkl"
    if cache_file.exists():
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    score_part = pt.load_musicxml(xml_path).parts[0]
    audio = generate_score_audio(score_part, tempo=tempo, samplerate=sr)
    with open(cache_file, "wb") as f:
        pickle.dump(audio, f, protocol=4)
    return audio


def get_cached_score_features(xml_path, tempo, processor_name="cqt", sr=SAMPLE_RATE):
    """CQT (or STFT) chroma of score audio. Cached."""
    key = _hash_inputs(*_file_mtime_size(xml_path), tempo, processor_name, sr)
    cache_file = CACHE_DIR / f"score_feats_{processor_name}_{key}.pkl"
    if cache_file.exists():
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    audio = get_cached_score_audio(xml_path, tempo, sr=sr)
    if processor_name == "cqt":
        from run_cqt_native import CQTChromagramProcessor
        proc = CQTChromagramProcessor(sample_rate=sr)
    else:
        from matchmaker.features.audio import ChromagramProcessor
        proc = ChromagramProcessor(sample_rate=sr)
    feats = proc(audio)
    with open(cache_file, "wb") as f:
        pickle.dump((audio, feats), f, protocol=4)
    return audio, feats


def get_cached_perf_audio(wav_path, sr=SAMPLE_RATE):
    """Load performance audio. Cached."""
    key = _hash_inputs(*_file_mtime_size(wav_path), sr)
    cache_file = CACHE_DIR / f"perf_audio_{key}.npy"
    if cache_file.exists():
        return np.load(cache_file)
    y, _ = librosa.load(wav_path, sr=sr)
    np.save(cache_file, y)
    return y


def get_cached_perf_features(wav_path, processor_name="cqt", sr=SAMPLE_RATE):
    """CQT chroma of performance audio. Cached."""
    key = _hash_inputs(*_file_mtime_size(wav_path), processor_name, sr)
    cache_file = CACHE_DIR / f"perf_feats_{processor_name}_{key}.pkl"
    if cache_file.exists():
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    y = get_cached_perf_audio(wav_path, sr=sr)
    if processor_name == "cqt":
        from run_cqt_native import CQTChromagramProcessor
        proc = CQTChromagramProcessor(sample_rate=sr)
    else:
        from matchmaker.features.audio import ChromagramProcessor
        proc = ChromagramProcessor(sample_rate=sr)
    feats = proc(y)
    with open(cache_file, "wb") as f:
        pickle.dump((y, feats), f, protocol=4)
    return y, feats


def cache_size_mb():
    total = sum(f.stat().st_size for f in CACHE_DIR.glob("*") if f.is_file())
    return total / 1024 / 1024


if __name__ == "__main__":
    print(f"Cache dir: {CACHE_DIR}")
    print(f"Cache size: {cache_size_mb():.1f} MB")
    files = sorted(CACHE_DIR.glob("*"))
    for f in files[:10]:
        print(f"  {f.name}: {f.stat().st_size/1024:.0f} KB")
    if len(files) > 10:
        print(f"  ... ({len(files)} total)")
