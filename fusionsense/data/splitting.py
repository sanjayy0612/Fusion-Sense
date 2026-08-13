"""Leakage-safe train/validation splitting for paired recordings.

Overlapping windows from one recording are strongly correlated. Splitting by
window would therefore put near-duplicates in train and validation. This
module keeps an entire recording together and, when the dataset has enough
subjects with complete class coverage, holds out whole subjects.
"""
from __future__ import annotations

from collections import defaultdict
import random


def _labels(windows):
    return {window.label for window in windows}


def _split_by_subject(windows, val_fraction, seed):
    grouped = defaultdict(list)
    for window in windows:
        if window.subject_id is None:
            return None
        grouped[window.subject_id].append(window)
    if len(grouped) < 2:
        return None

    subjects = sorted(grouped)
    random.Random(seed).shuffle(subjects)
    n_val = max(1, round(len(subjects) * val_fraction))
    n_val = min(n_val, len(subjects) - 1)
    val_subjects = set(subjects[:n_val])
    train = [w for w in windows if w.subject_id not in val_subjects]
    val = [w for w in windows if w.subject_id in val_subjects]
    if _labels(train) == _labels(windows) and _labels(val) == _labels(windows):
        return train, val, f"held-out subjects: {', '.join(sorted(val_subjects))}"
    return None


def _split_by_recording(windows, val_fraction, seed):
    grouped = defaultdict(list)
    for index, window in enumerate(windows):
        key = window.recording_id or f"window-{index}"
        grouped[(window.label, key)].append(window)

    rng = random.Random(seed)
    train_keys = set()
    val_keys = set()
    by_label = defaultdict(list)
    for label, key in grouped:
        by_label[label].append((label, key))

    for keys in by_label.values():
        keys.sort(key=lambda item: item[1])
        rng.shuffle(keys)
        # A class needs at least two recordings to appear in both partitions.
        if len(keys) < 2:
            raise ValueError(
                "Each activity needs at least two separate recordings before "
                "fusion training; collect another trial for every activity."
            )
        n_val = max(1, round(len(keys) * val_fraction))
        n_val = min(n_val, len(keys) - 1)
        val_keys.update(keys[:n_val])
        train_keys.update(keys[n_val:])

    train = []
    val = []
    for group_key, group_windows in grouped.items():
        (val if group_key in val_keys else train).extend(group_windows)
    return train, val, "stratified by recording"


def split_paired_windows(windows, val_fraction=0.2, seed=42):
    """Return ``(train, validation, description)`` without recording leakage."""
    windows = list(windows)
    if not windows:
        raise ValueError("Cannot split an empty paired dataset")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")

    subject_split = _split_by_subject(windows, val_fraction, seed)
    if subject_split is not None:
        return subject_split
    return _split_by_recording(windows, val_fraction, seed)
