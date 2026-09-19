from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "datasets"
DEFAULT_DATASET_ID = "20260213"


@dataclass(frozen=True)
class DatasetConfig:
    dataset_id: str
    display_name: str
    raw_root: Path
    raw_kind: str
    position_glob: str | None
    primary_timelapse_czi: Path | None
    auxiliary_czi_files: tuple[Path, ...]
    scene_name_parser: str | None
    frame_interval_min: float | None
    channel_names: tuple[str, ...]
    ilastik_channel_indices_v1: tuple[int, ...]
    ilastik_root: Path
    project_name_v1: str
    training_set_name_v1: str
    probability_class_names_v1: tuple[str, ...]
    notes: str
    training_stacks_v1: tuple[dict[str, Any], ...]
    scene_block_conditions: tuple[dict[str, Any], ...]
    position_condition_file: Path | None
    position_condition_format: str | None
    uniform_media_condition: str | None
    exclude_position_labels_batch_v1: tuple[str, ...]

    @property
    def dataset_results_root(self) -> Path:
        return ROOT / "results" / "datasets" / self.dataset_id


def available_dataset_ids() -> list[str]:
    return sorted(p.stem for p in CONFIG_DIR.glob("*.json"))


def resolve_dataset_id(dataset_id: str | None = None) -> str:
    return dataset_id or os.environ.get("FOXF1_DATASET_ID", DEFAULT_DATASET_ID)


def _load_json(dataset_id: str) -> dict[str, Any]:
    path = CONFIG_DIR / f"{dataset_id}.json"
    if not path.exists():
        known = ", ".join(available_dataset_ids())
        raise FileNotFoundError(f"Missing dataset config for {dataset_id}. Known dataset IDs: {known}")
    return json.loads(path.read_text())


def load_dataset_config(dataset_id: str | None = None) -> DatasetConfig:
    dataset_id = resolve_dataset_id(dataset_id)
    raw = _load_json(dataset_id)
    return DatasetConfig(
        dataset_id=raw["dataset_id"],
        display_name=raw["display_name"],
        raw_root=ROOT / raw["raw_root_rel"],
        raw_kind=raw["raw_kind"],
        position_glob=raw.get("position_glob"),
        primary_timelapse_czi=(ROOT / raw["primary_timelapse_czi_rel"]) if raw.get("primary_timelapse_czi_rel") else None,
        auxiliary_czi_files=tuple(ROOT / rel for rel in raw.get("auxiliary_czi_file_rels", [])),
        scene_name_parser=raw.get("scene_name_parser"),
        frame_interval_min=raw.get("frame_interval_min"),
        channel_names=tuple(raw.get("channel_names", [])),
        ilastik_channel_indices_v1=tuple(raw.get("ilastik_channel_indices_v1", [0, 1])),
        ilastik_root=ROOT / raw["ilastik_root_rel"],
        project_name_v1=raw.get("project_name_v1", "pixelclass_v1.ilp"),
        training_set_name_v1=raw.get("training_set_name_v1", "phase_rfp_curated_v1"),
        probability_class_names_v1=tuple(raw.get("probability_class_names_v1", [])),
        notes=raw.get("notes", ""),
        training_stacks_v1=tuple(raw.get("training_stacks_v1", [])),
        scene_block_conditions=tuple(raw.get("scene_block_conditions", [])),
        position_condition_file=(ROOT / raw["position_condition_file_rel"]) if raw.get("position_condition_file_rel") else None,
        position_condition_format=raw.get("position_condition_format"),
        uniform_media_condition=raw.get("uniform_media_condition"),
        exclude_position_labels_batch_v1=tuple(raw.get("exclude_position_labels_batch_v1", [])),
    )


def ensure_dataset_scaffold(dataset: DatasetConfig) -> None:
    dataset.dataset_results_root.mkdir(parents=True, exist_ok=True)
    (dataset.dataset_results_root / "metadata").mkdir(parents=True, exist_ok=True)
    for subdir in ["projects", "training_frames", "qc", "notes", "batch_processing_data_v1/inputs", "batch_processing_data_v1/probabilities"]:
        (dataset.ilastik_root / subdir).mkdir(parents=True, exist_ok=True)


def dataset_summary_lines(dataset: DatasetConfig) -> list[str]:
    lines = [
        f"- dataset_id: `{dataset.dataset_id}`",
        f"- display_name: `{dataset.display_name}`",
        f"- raw_root: `{dataset.raw_root}`",
        f"- raw_kind: `{dataset.raw_kind}`",
        f"- ilastik_root: `{dataset.ilastik_root}`",
        f"- project_name_v1: `{dataset.project_name_v1}`",
    ]
    if dataset.probability_class_names_v1:
        lines.append(f"- probability_class_names_v1: `{', '.join(dataset.probability_class_names_v1)}`")
    if dataset.primary_timelapse_czi is not None:
        lines.append(f"- primary_timelapse_czi: `{dataset.primary_timelapse_czi}`")
    if dataset.auxiliary_czi_files:
        lines.append(f"- auxiliary_czi_files: `{'; '.join(str(p) for p in dataset.auxiliary_czi_files)}`")
    if dataset.scene_name_parser is not None:
        lines.append(f"- scene_name_parser: `{dataset.scene_name_parser}`")
    if dataset.notes:
        lines.append(f"- notes: {dataset.notes}")
    if dataset.scene_block_conditions:
        lines.append(f"- scene_block_conditions: `{len(dataset.scene_block_conditions)}` block specs")
    if dataset.position_condition_file is not None:
        lines.append(f"- position_condition_file: `{dataset.position_condition_file}`")
    if dataset.uniform_media_condition is not None:
        lines.append(f"- uniform_media_condition: `{dataset.uniform_media_condition}`")
    if dataset.exclude_position_labels_batch_v1:
        lines.append(
            f"- exclude_position_labels_batch_v1: `{', '.join(dataset.exclude_position_labels_batch_v1)}`"
        )
    return lines


def parse_position_index(position_label: str) -> int | None:
    if re.fullmatch(r"Pos\d+", position_label):
        return int(position_label[3:])
    m = re.match(r"^\d+-Pos(\d{3})_(\d{3})$", position_label)
    if m:
        return None
    return None


def load_position_condition_map(dataset: DatasetConfig) -> dict[Any, dict[str, Any]]:
    if dataset.position_condition_file is None:
        return {}
    if dataset.position_condition_format == "pos_index_ranges":
        text = dataset.position_condition_file.read_text()
        mapping: dict[Any, dict[str, Any]] = {}
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            m = re.match(r"^(\d+)\s*-\s*(\d+)\s+(.*)$", line)
            if not m:
                continue
            start = int(m.group(1))
            end = int(m.group(2))
            label = m.group(3).strip()
            if "(" in label and label.endswith(")"):
                base, note = label.rsplit("(", 1)
                condition_label = base.strip()
                condition_note = note[:-1].strip()
            else:
                condition_label = label
                condition_note = ""
            for pos_idx in range(start, end + 1):
                mapping[pos_idx] = {
                    "position_index": pos_idx,
                    "media_condition": condition_label,
                    "media_condition_note": condition_note,
                    "media_condition_source": str(dataset.position_condition_file),
                    "media_condition_range": f"{start}-{end}",
                }
        return mapping

    if dataset.position_condition_format == "position_label_table":
        rows = _load_condition_table(dataset.position_condition_file)
        mapping: dict[Any, dict[str, Any]] = {}
        for row in rows:
            label = str(row.get("position_label", "")).strip()
            if not label:
                continue
            row["media_condition_source"] = str(dataset.position_condition_file)
            mapping[label] = row
            pos_idx = parse_position_index(label)
            if pos_idx is not None and pos_idx not in mapping:
                mapping[pos_idx] = row
        return mapping

    raise NotImplementedError(
        f"Unsupported position_condition_format={dataset.position_condition_format!r} "
        f"for dataset {dataset.dataset_id}"
    )


def lookup_position_condition(
    position_condition_map: dict[Any, dict[str, Any]],
    *,
    position_label: str,
    position_index: int | None,
) -> dict[str, Any]:
    label = position_label.strip()
    if label and label in position_condition_map:
        return position_condition_map[label]
    if position_index is not None and position_index in position_condition_map:
        return position_condition_map[position_index]
    return {}


def _load_condition_table(path: Path) -> Any:
    suffix = path.suffix.lower()
    if suffix == ".tsv":
        import csv

        with path.open(newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            return list(reader)
    if suffix == ".csv":
        import csv

        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            return list(reader)

    raise NotImplementedError(f"Unsupported condition table format for {path}")
