from __future__ import annotations

import re
from pathlib import Path
from typing import Any

# This script needs czifile's `filtered_subblock_directory`, which exposes czifile's own
# subblock objects and has no libCZI equivalent, so it cannot use src/trunk_morph_ref/czi_compat.
# czifile itself is no longer installable beside this project's pinned numpy: the 2019 release
# calls imagecodecs.jxr_decode (renamed jpegxr_decode) and the current release requires numpy 2.x.
# This mosaic step is therefore NOT runnable as shipped; its outputs are committed under derived/.
# The import is DEFERRED into _czifile() rather than taken at module scope so that importing
# this module still works: several runnable scripts import it for its other helpers, and a
# top-level import here made every one of them fail too.
def _czifile():
    """Import czifile on demand, with a usable message when it cannot be installed."""
    try:
        import czifile
    except Exception as exc:                                        # noqa: BLE001
        raise RuntimeError(
            "This mosaic step needs czifile, which cannot be installed beside this project's "
            "pinned numpy (see env/environment.yml). Its outputs are committed under derived/."
        ) from exc
    return czifile
import numpy as np


def _ensure_list(obj: Any) -> list[Any]:
    if obj is None:
        return []
    if isinstance(obj, list):
        return obj
    return [obj]


def extract_channels(metadata: dict[str, Any], c_count: int | None = None) -> list[str]:
    dims = (
        metadata.get("ImageDocument", {})
        .get("Metadata", {})
        .get("Information", {})
        .get("Image", {})
        .get("Dimensions", {})
    )
    channel_obj = dims.get("Channels", {}).get("Channel")
    channels = _ensure_list(channel_obj)
    names: list[str] = []
    for idx, ch in enumerate(channels):
        if isinstance(ch, dict):
            names.append(
                str(
                    ch.get("ShortName")
                    or ch.get("Name")
                    or ch.get("DyeName")
                    or f"Channel_{idx}"
                )
            )
        else:
            names.append(f"Channel_{idx}")
    if c_count is not None and len(names) < int(c_count):
        names.extend(f"Channel_{idx}" for idx in range(len(names), int(c_count)))
    return names[: int(c_count)] if c_count is not None else names


def extract_scale_um(metadata: dict[str, Any]) -> dict[str, float]:
    scale = {"X": np.nan, "Y": np.nan, "Z": np.nan}
    items = (
        metadata.get("ImageDocument", {})
        .get("Metadata", {})
        .get("Scaling", {})
        .get("Items", {})
        .get("Distance")
    )
    for item in _ensure_list(items):
        if not isinstance(item, dict):
            continue
        axis = str(item.get("Id", "")).upper()
        value = item.get("Value")
        try:
            um = float(value) * 1e6
        except Exception:
            continue
        if axis in scale:
            scale[axis] = um
    return scale


def extract_objective_name(metadata: dict[str, Any]) -> str:
    objective = (
        metadata.get("ImageDocument", {})
        .get("Metadata", {})
        .get("Information", {})
        .get("Image", {})
        .get("ObjectiveSettings", {})
    )
    if isinstance(objective, dict):
        return str(objective.get("ObjectiveRef", "") or objective.get("Name", ""))
    return ""


def get_scene_records(path: Path, scene_name_parser: str | None = None) -> list[dict[str, Any]]:
    with _czifile().CziFile(path) as czi:
        metadata = czi.metadata(raw=False)
        axes = str(czi.axes)
        shape = tuple(int(x) for x in czi.shape)

    image = (
        metadata.get("ImageDocument", {})
        .get("Metadata", {})
        .get("Information", {})
        .get("Image", {})
    )
    dims = image.get("Dimensions", {})
    scene_obj = dims.get("S", {}).get("Scenes", {}).get("Scene")
    scenes = _ensure_list(scene_obj)

    axis_idx = {ax: i for i, ax in enumerate(axes)}
    size_t = shape[axis_idx["T"]] if "T" in axis_idx else 1
    size_c = shape[axis_idx["C"]] if "C" in axis_idx else 1
    channels = extract_channels(metadata, c_count=size_c)

    out: list[dict[str, Any]] = []
    for idx, scene in enumerate(scenes):
        if not isinstance(scene, dict):
            continue
        name = str(scene.get("Name", "") or f"Scene{idx:02d}")
        center = str(scene.get("CenterPosition", "") or "")
        x_um = np.nan
        y_um = np.nan
        if center and "," in center:
            try:
                x_str, y_str = [part.strip() for part in center.split(",", 1)]
                x_um = float(x_str)
                y_um = float(y_str)
            except Exception:
                pass

        parsed = parse_scene_name(name, parser_key=scene_name_parser)
        out.append(
            {
                "scene_index": int(scene.get("Index", idx)),
                "scene_name_raw": name,
                "scene_region_id": scene.get("RegionId", ""),
                "scene_scan_mode": scene.get("ScanMode", ""),
                "center_x_um": x_um,
                "center_y_um": y_um,
                "size_t": int(size_t),
                "size_c": int(size_c),
                "channel_names": ";".join(channels),
                **parsed,
            }
        )
    return out


def normalize_scene_label(scene_index: int, scene_name_raw: str, dataset_id: str | None = None) -> str:
    prefix = f"{dataset_id}_" if dataset_id else ""
    safe_name = re.sub(r"[^A-Za-z0-9]+", "_", scene_name_raw).strip("_")
    return f"{prefix}S{int(scene_index):02d}_{safe_name}"


def parse_scene_name(scene_name_raw: str, parser_key: str | None = None) -> dict[str, Any]:
    if parser_key in (None, "", "identity"):
        return {
            "scene_group": scene_name_raw,
            "media_condition": scene_name_raw,
            "replicate_label": "",
            "replicate_index": np.nan,
        }
    if parser_key == "20260314_embedded_conditions":
        return _parse_scene_name_20260314(scene_name_raw)
    raise NotImplementedError(f"Unsupported scene_name_parser={parser_key!r}")


def _parse_scene_name_20260314(scene_name_raw: str) -> dict[str, Any]:
    name = str(scene_name_raw).strip()
    m = re.match(r"^(.*?)([Pp])(\d+)$", name)
    if m:
        base = m.group(1)
        rep_label = f"P{int(m.group(3))}"
        rep_index = int(m.group(3))
    else:
        base = name
        rep_label = ""
        rep_index = np.nan

    key = re.sub(r"[^a-z0-9]", "", base.lower())
    mapping = {
        "19hrd1": "19hr d1",
        "19hldn": "19hr LDN",
        "19hbasal": "19hr basal",
        "24hbasal": "24hr basal",
        "24d1": "24hr d1",
        "24ldn": "24hr LDN",
        "15basal": "15hr basal",
        "15hldn": "15hr LDN",
        "19ldnw2": "19hr LDN w2",
        "19basalw2": "19hr basal w2",
        "19basalw3": "19hr basal w3",
        "19ldnw3": "19hr LDN w3",
    }
    media_condition = mapping.get(key, base)
    return {
        "scene_group": base,
        "media_condition": media_condition,
        "replicate_label": rep_label,
        "replicate_index": rep_index,
    }


def extract_scene_stack(
    czi_path: Path,
    scene_index: int,
    channel_indices: tuple[int, ...] = (0, 1),
    t_start: int | None = None,
    t_end: int | None = None,
) -> np.ndarray:
    with _czifile().CziFile(czi_path) as czi:
        axes = str(czi.axes)
        shape = tuple(int(v) for v in czi.shape)
        fsbd = czi.filtered_subblock_directory

        axis_idx = {ax: i for i, ax in enumerate(axes)}
        if not {"S", "T", "C"}.issubset(axis_idx):
            raise RuntimeError(f"Expected CZI axes to include S,T,C; got {axes}")
        size_t = int(shape[axis_idx["T"]])
        t0 = 0 if t_start is None else int(t_start)
        t1 = size_t - 1 if t_end is None else int(t_end)
        if t0 < 0 or t1 >= size_t or t0 > t1:
            raise ValueError(f"Invalid t_start/t_end for {czi_path.name}: {t_start}, {t_end}")

        tile_map: dict[tuple[int, int], np.ndarray] = {}
        tile_shape: tuple[int, int] | None = None
        for d in fsbd:
            start = d.start
            s_idx = int(start[axis_idx["S"]])
            if s_idx != int(scene_index):
                continue
            t_idx = int(start[axis_idx["T"]])
            if t_idx < t0 or t_idx > t1:
                continue
            c_idx = int(start[axis_idx["C"]])
            if c_idx not in channel_indices:
                continue
            tile = np.asarray(d.data_segment().data(raw=False, resize=True), dtype=np.uint16)
            tile = np.squeeze(tile)
            if tile.ndim != 2:
                raise RuntimeError(
                    f"Expected 2D tile for scene={scene_index}, time={t_idx}, channel={c_idx}; got {tile.shape}"
                )
            tile_map[(t_idx, c_idx)] = tile
            if tile_shape is None:
                tile_shape = tuple(int(v) for v in tile.shape)

    if tile_shape is None:
        raise RuntimeError(f"No matching tiles found in {czi_path} for scene {scene_index}")

    times = list(range(t0, t1 + 1))
    stack = np.zeros((len(times), len(channel_indices), tile_shape[0], tile_shape[1]), dtype=np.uint16)
    for t_pos, t_idx in enumerate(times):
        for c_pos, c_idx in enumerate(channel_indices):
            key = (t_idx, c_idx)
            if key not in tile_map:
                raise RuntimeError(
                    f"Missing tile in {czi_path.name} for scene={scene_index}, time={t_idx}, channel={c_idx}"
                )
            stack[t_pos, c_pos] = tile_map[key]
    return stack
