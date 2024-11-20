#!/usr/bin/env python3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple, TypeAlias

import numpy as np
import open3d as o3d
import pandas as pd
import torch
import yaml

PuWo: TypeAlias = o3d.geometry.PointCloud

DATA_PATH = Path(__file__) / ".." / ".." / "data"


@dataclass(slots=True, kw_only=True)
class DirPathConfig:
    directory: str
    search: str

    @property
    def dir_path(self) -> Path:
        return Path(self.directory)


@dataclass(slots=True, kw_only=True)
class FilesConfig:
    puwo: DirPathConfig
    labels: DirPathConfig


@dataclass(slots=True, kw_only=True)
class TrainingConfig:
    batch_size: int
    num_epochs: int
    learning_rate: float


@dataclass(slots=True, kw_only=True)
class FilterCionfig:
    default_radius: float
    enabled: bool


@dataclass(slots=True, kw_only=True)
class AiConfig:
    filtering: FilterCionfig
    files: FilesConfig


def _families_path() -> Path:
    return DATA_PATH / "raw" / "families"


def _config_path(name: str) -> Path:
    return DATA_PATH / ".." / "config" / f"{name}.yaml"


def load_labal_n_segment(file_path: Path) -> pd.DataFrame:
    return pd.read_csv(file_path.resolve())


def load_label_n_segments(file_path: Path) -> pd.DataFrame:
    data_frame = load_labal_n_segment(file_path)
    return data_frame


def filter_points_by_radius(point_cloud: PuWo, center: Tuple[float, float], radius: float) -> PuWo:
    points = np.asarray(point_cloud.points)
    mask = np.linalg.norm(points - center, axis=1) <= radius
    filtered_cloud = point_cloud.select_by_index(np.where(mask)[0])
    return filtered_cloud


def load_ply(file_path: Path) -> np.ndarray:
    point_cloud = o3d.io.read_point_cloud(str(file_path.resolve()))
    return np.asarray(point_cloud.points)


def point_cloud_to_tensor(point_cloud: PuWo) -> torch.Tensor:
    points = np.asarray(point_cloud.points)
    return torch.tensor(points, dtype=torch.float32)


def read_processing_config() -> AiConfig:
    with open(_config_path("preprocessing_config"), "r") as yaml_file:
        config = yaml.safe_load(yaml_file)
    return AiConfig(**config)


def get_puwo_file_paths(config: AiConfig) -> Iterable[Path]:
    puwo_config = config.files.puwo
    dir_path = puwo_config.dir_path
    for file_path in dir_path.glob(puwo_config.search):
        yield file_path.resolve()


def get_label_file_paths(config: AiConfig) -> Iterable[Path]:
    label_config = config.files.labels
    dir_path = label_config.dir_path
    for file_path in dir_path.glob(label_config.search):
        yield file_path.resolve()


# Lade alle Dateien
config = read_processing_config()
ply_files = get_puwo_file_paths(config)
label_files = get_label_file_paths(config)


# Initialisiere leere Listen für kombinierte Daten
elem_points = []
meta_values = []
point_segments = []

# Schleife über alle Punktwolken- und Labels-Dateien
for ply_file, meta_file in zip(ply_files, label_files):
    # Punktwolke einlesen
    points = load_ply(file_path=ply_file)
    meta_data = load_label_n_segments(file_path=meta_file)

    # Prüfen, ob die Anzahl der Punkte und Labels übereinstimmt
    if len(points) != len(meta_data):
        raise ValueError(
            f"{ply_file}: Anzahl Punkte '{len(points)}', Metadaten '{len(meta_data)}' stimmen nicht übewrein"
        )

    coord = meta_data["coordinate"]
    if coord is not None and not coord.empty:
        center = np.array(coord)
        radius = config.filtering.default_radius
        mask = np.linalg.norm(points - center, axis=1) <= radius
        points = points[mask]
        meta_data = meta_data[mask]

    # Füge die Daten zur Gesamtliste hinzu
    elem_points.append(points)
    meta_values.append(meta_data)

# Kombiniere alle Punktwolken und Labels
# elem_points = np.vstack(elem_points)
# meta_values = np.concatenate(meta_values)
# point_segments = np.concatenate(point_segments)

# Speichere die kombinierten Daten im NPY-Format
np.save("data/processed/train_points.npy", elem_points)
np.save("data/processed/train_labels.npy", meta_values)
np.save("data/processed/train_seqments.npy", point_segments)
