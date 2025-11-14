"""插件注册与获取"""

from __future__ import annotations

from typing import Dict, Type

from .base import Plugin, ExamplePlugin


_REGISTRY: Dict[str, Plugin] = {}


def register_default_plugins() -> None:
    """中文说明：注册内置/示例插件；实际项目可根据依赖条件注册。"""
    _REGISTRY["example"] = ExamplePlugin()


def get_plugin(name: str) -> Plugin:
    if not _REGISTRY:
        register_default_plugins()
    return _REGISTRY[name]