
import open3d as o3d
import numpy as np
import pandas as pd
import yaml

# Lade die Konfiguration
with open("config/preprocessing_config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Initialisiere leere Listen für kombinierte Daten
all_points = []
all_labels = []

# Schleife über alle Punktwolken- und Labels-Dateien
for ply_file, label_file in zip(config["files"]["point_clouds"], config["files"]["labels"]):
    # Punktwolke einlesen
    point_cloud = o3d.io.read_point_cloud(ply_file)
    points = np.asarray(point_cloud.points)

    # Labels laden
    labels_df = pd.read_csv(label_file)
    segment_ids = labels_df["segment_id"].values

    # Prüfen, ob die Anzahl der Punkte und Labels übereinstimmt
    if len(points) != len(segment_ids):
        raise ValueError(f"Anzahl der Punkte und Labels stimmt nicht überein in {ply_file}!")

    # Radiusfilterung (falls aktiviert)
    if config["radius_filtering"]["enabled"]:
        center = np.array([0.5, 0.5, 0.0])  # Beispielkoordinate
        radius = config["radius_filtering"]["default_radius"]
        mask = np.linalg.norm(points - center, axis=1) <= radius
        points = points[mask]
        segment_ids = segment_ids[mask]  # Labels entsprechend filtern

    # Füge die Daten zur Gesamtliste hinzu
    all_points.append(points)
    all_labels.append(segment_ids)

# Kombiniere alle Punktwolken und Labels
all_points = np.vstack(all_points)
all_labels = np.concatenate(all_labels)

# Speichere die kombinierten Daten im NPY-Format
np.save("data/processed/train_points.npy", all_points)
np.save("data/processed/train_labels.npy", all_labels)
