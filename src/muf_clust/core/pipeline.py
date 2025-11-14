"""统一管线与步骤抽象

中文说明：
- Step：封装单个阶段的处理；Pipeline：顺序编排与上下文传递。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Protocol

from ..utils.logging import log_info, log_error


class Step(Protocol):
    name: str

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        ...


@dataclass
class Pipeline:
    steps: List[Step]

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """顺序执行步骤，聚合结果并返回最终上下文。

        中文说明：
        - 每步可返回更新内容，合并到 context 中。
        - 遇到异常立即记录并抛出，避免静默失败。
        """
        log_info(f"启动管线，共 {len(self.steps)} 个步骤")
        for step in self.steps:
            try:
                log_info(f"执行步骤：{getattr(step, 'name', step.__class__.__name__)}")
                update = step.run(context)
                if isinstance(update, dict):
                    context.update(update)
            except Exception as e:
                log_error(f"步骤执行失败：{step.__class__.__name__}，错误：{e}")
                raise
        log_info("管线执行完成")
        return context