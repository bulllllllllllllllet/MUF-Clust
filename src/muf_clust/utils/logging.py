"""统一日志工具

中文说明：
- 控制台使用分隔标记，提升可读性；绘图保持英文；调试中文。
"""

from __future__ import annotations

import sys
from typing import Any


def _block(msg: str) -> str:
    return f"\n_____\n{msg}\n_____\n"


def _print(level: str, msg: str) -> None:
    sys.stdout.write(_block(f"[{level}] {msg}"))
    sys.stdout.flush()


def log_info(msg: str) -> None:
    """中文：信息级别日志。"""
    _print("INFO", msg)


def log_debug(msg: str) -> None:
    """中文：调试级别日志。"""
    _print("DEBUG", msg)


def log_warn(msg: str) -> None:
    """中文：警告级别日志。"""
    _print("WARN", msg)


def log_error(msg: str) -> None:
    """中文：错误级别日志。"""
    _print("ERROR", msg)