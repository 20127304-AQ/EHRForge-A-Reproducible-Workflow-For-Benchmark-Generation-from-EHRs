"""Canonical sampled_window reconstruction (frozen in shared/canonical/protocol.yaml, Step 0).
Validated in Step 0: for all 24,141 evidence items, original_visit_index is in window(strategy, V)
and its position in the window equals the chunk-relative visit_index."""
import re

def window(strategy: str, V: int):
    m = re.fullmatch(r"local_chunk_(\d+)_(\d+)_size_(\d+)_overlap_(\d+)", strategy)
    if m:
        a, b = int(m.group(1)), int(m.group(2)); return list(range(a, min(b, V)))
    m = re.fullmatch(r"(?:global_)?u_shape_first_(\d+)_last_(\d+)", strategy)
    if m:
        f, l = int(m.group(1)), int(m.group(2))
        if f + l >= V: return list(range(V))
        return list(range(f)) + list(range(V - l, V))
    m = re.fullmatch(r"full_timeline_(\d+)_(\d+)", strategy)
    if m:
        return list(range(V))
    raise ValueError(f"unknown timeline_sampling_strategy: {strategy}")

