"""路径与命名工具（对齐 guide.md 5.4）"""

from __future__ import annotations

import os


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def qc_dir(base_out: str, sample_id: str) -> str:
    d = os.path.join(base_out, sample_id, "qc")
    return ensure_dir(d)


def prefixed_name(sample_id: str, index: int, name: str) -> str:
    return f"{sample_id}_{index:02d}_{name}.png"