"""通道配置（根据 guide.md）

提供两套默认配置：肾癌（KidneyCancer）与膀胱癌（BladderCancer）。
index 字段表示通道编号，is_dapi / is_af 用于识别特殊通道。

注意：实际 qptiff 中的通道顺序与命名可能不同，建议后续从元数据解析。
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
import os
try:
    import yaml  # type: ignore
except Exception:
    yaml = None  # type: ignore


@dataclass
class ChannelSpec:
    index: int
    name: str
    marker: str
    biological_role: str
    is_dapi: bool = False
    is_af: bool = False


CHANNEL_CONFIGS: Dict[str, List[ChannelSpec]] = {
    "KidneyCancer": [
        ChannelSpec(1, "DAPI", "-", "核染色", is_dapi=True),
        ChannelSpec(2, "Opal480", "CD20", "细胞膜"),
        ChannelSpec(3, "Opal520", "CD3", "细胞膜"),
        ChannelSpec(4, "Opal570", "FOXP3", "细胞核"),
        ChannelSpec(5, "Opal620", "CD23", "细胞膜"),
        ChannelSpec(6, "Opal690", "CD8", "细胞膜"),
        ChannelSpec(7, "Opal780", "PAX8s", "细胞核"),
        ChannelSpec(8, "SampleAF", "AF", "细胞自发荧光（背景）", is_af=True),
    ],
    "BladderCancer": [
        ChannelSpec(1, "DAPI", "-", "核染色", is_dapi=True),
        ChannelSpec(2, "Opal480", "PANCK", "细胞质"),
        ChannelSpec(3, "Opal520", "IgM", "细胞膜"),
        ChannelSpec(4, "Opal570", "FOXP3", "细胞核"),
        ChannelSpec(5, "Opal620", "CD4", "细胞膜"),
        ChannelSpec(6, "Opal690", "PD-L1", "细胞膜"),
        ChannelSpec(7, "Opal780", "CXCL13", "细胞核"),
        ChannelSpec(8, "SampleAF", "AF", "细胞自发荧光（背景）", is_af=True),
    ],
}


DEFAULT_CANCER_TYPE = "KidneyCancer"


@dataclass
class RuntimeDefaults:
    dataset_dir: Optional[str] = None
    image_path: Optional[str] = None
    output_root: str = "outputs/preprocess"
    tile_size: int = 512
    seed: int = 42


RUNTIME_DEFAULTS: Dict[str, RuntimeDefaults] = {
    "KidneyCancer": RuntimeDefaults(
        dataset_dir="/nfs5/yj/MIHC/dataset/KidneyCancer",
        output_root="outputs/preprocess",
        tile_size=512,
        seed=42,
    ),
    "BladderCancer": RuntimeDefaults(
        dataset_dir="/nfs5/yj/MIHC/dataset/BladderCancer",
        output_root="outputs/preprocess",
        tile_size=512,
        seed=42,
    ),
}


def _channels_yaml_path(cancer_type: str) -> Optional[str]:
    base = os.path.join(os.path.dirname(__file__), "configs")
    mapping = {
        "KidneyCancer": "channels_kidney.yaml",
        "BladderCancer": "channels_bladder.yaml",
    }
    fn = mapping.get(cancer_type)
    if not fn:
        return None
    path = os.path.join(base, fn)
    return path if os.path.isfile(path) else None


def _load_channels_from_yaml(path: str) -> Optional[List[ChannelSpec]]:
    if yaml is None:
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    items = []
    if isinstance(data, list):
        for it in data:
            try:
                items.append(ChannelSpec(
                    index=int(it.get("index")),
                    name=str(it.get("name")),
                    marker=str(it.get("marker", "")),
                    biological_role=str(it.get("biological_role", "")),
                    is_dapi=bool(it.get("is_dapi", False)),
                    is_af=bool(it.get("is_af", False)),
                ))
            except Exception:
                continue
    return items if items else None


def get_config(cancer_type: str) -> Optional[List[ChannelSpec]]:
    yp = _channels_yaml_path(cancer_type)
    if yp:
        loaded = _load_channels_from_yaml(yp)
        if loaded:
            return loaded
    return CHANNEL_CONFIGS.get(cancer_type)


def get_runtime_defaults(cancer_type: Optional[str]) -> RuntimeDefaults:
    key = cancer_type or DEFAULT_CANCER_TYPE
    return RUNTIME_DEFAULTS.get(key, RuntimeDefaults())