#!/usr/bin/env python3
"""Convert HCP Movie 1 WordNet annotations into Bridgeville content features.

The HCP-provided WordNet annotation file contains one annotation record per
movie time point. Bridgeville corresponds to annotation IDs 284-505 (222 TRs)
in the study alignment. A feature is coded 1 when any WordNet label assigned to
that feature is present at a TR, otherwise 0.

The six primary downstream features are human presence, communication, face
attention, active movement, music performance, and road/vehicle presence.
Outdoor scene and on-screen text are used as nuisance regressors. Static posture
and indoor scene are retained in the exported feature table for annotation QC
but are not part of the primary second-level test family.

The HCP annotation file itself is not distributed in this repository.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

BRIDGEVILLE_START = 284
BRIDGEVILLE_END = 505
TR_TO_ANNOTATION_OFFSET = 284

HUMAN = {
    "man.n.01", "woman.n.01", "girl.n.01", "male_child.n.01",
    "female_child.n.01", "old_man.n.01", "old_woman.n.01", "people.n.01",
    "football_player.n.01", "footballer.n.01", "racer.n.02", "boy_scout.n.01",
}
COMMUNICATION = {"talk.v.02", "gesticulate.v.01", "sing.v.01"}
FACE_ATTENTION = {
    "face.n.01", "gaze.v.01", "look.v.01", "look.v.02", "smile.v.01",
    "smile.n.01", "blink.v.01",
}
ACTIVE_MOVEMENT = {
    "run.v.01", "walk.v.01", "drive.v.16", "drive.v.01", "play.v.01",
    "play.v.03", "play.v.05", "mow.v.01", "weld.v.01", "barber.v.01",
    "cut.v.32", "box.v.02", "roll.v.01", "push.v.01",
}
STATIC_POSTURE = {"sit.v.01", "stand.v.01"}
MUSIC_PERFORMANCE = {
    "music_stand.n.01", "musical_instrument.n.01", "play.v.03", "sing.v.01",
    "sax.n.02", "band.n.02",
}
VEHICLE = {"car.n.01", "pickup.n.01", "sports_car.n.01", "fire_engine.n.01", "truck.n.01"}
OUTDOOR = {
    "sky.n.01", "tree.n.01", "road.n.01", "lawn.n.01", "cloud.n.02",
    "sidewalk.n.01", "neighborhood.n.04", "plant.n.02",
}
INDOOR = {
    "room.n.01", "wall.n.01", "wall.n.02", "window.n.01", "workshop.n.01",
    "library.n.01", "library.n.02", "barbershop.n.01", "showroom.n.01",
    "locker_room.n.01", "garage.n.01", "garage.n.02", "living_room.n.01",
    "hall.n.03",
}
TEXT = {"subtitle.n.01", "text.n.01", "title.n.08", "credits.n.01"}

FEATURES = {
    "human_presence": HUMAN,
    "communication": COMMUNICATION,
    "face_attention": FACE_ATTENTION,
    "active_movement": ACTIVE_MOVEMENT,
    "static_posture": STATIC_POSTURE,
    "music_performance": MUSIC_PERFORMANCE,
    "road_vehicle_presence": VEHICLE,
    "outdoor_scene": OUTDOOR,
    "indoor_scene": INDOOR,
    "on_screen_text": TEXT,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True, help="HCP WordNet_MOVIE1_CC1.txt")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_annotations(path: Path) -> dict[int, set[str]]:
    annotations: dict[int, set[str]] = {}
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            match = re.match(r"(\d+):", line)
            if match is None:
                continue
            annotation_id = int(match.group(1))
            if BRIDGEVILLE_START <= annotation_id <= BRIDGEVILLE_END:
                annotations[annotation_id] = set(re.findall(r'"([^"]+)"', line))
    return annotations


def main() -> None:
    args = parse_args()
    annotations = load_annotations(args.annotations)
    n_trs = BRIDGEVILLE_END - BRIDGEVILLE_START + 1

    rows = []
    for brain_tr in range(n_trs):
        annotation_id = brain_tr + TR_TO_ANNOTATION_OFFSET
        labels = annotations.get(annotation_id, set())
        row = {
            "brain_TR": brain_tr,
            "annotation_ID": annotation_id,
            "n_labels": len(labels),
        }
        for feature_name, feature_labels in FEATURES.items():
            row[feature_name] = int(bool(labels.intersection(feature_labels)))
        rows.append(row)

    features = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.output, index=False)

    print(f"Bridgeville TRs: {n_trs}")
    print("\nFeature counts")
    for feature in FEATURES:
        count = int(features[feature].sum())
        print(f"{feature:24s} {count:3d} TRs ({100 * features[feature].mean():5.1f}%)")

    assigned_labels = set().union(*FEATURES.values())
    total_occurrences = assigned_occurrences = 0
    trs_with_labels = trs_with_assigned = 0
    for annotation_id in range(BRIDGEVILLE_START, BRIDGEVILLE_END + 1):
        labels = annotations.get(annotation_id, set())
        assigned = labels.intersection(assigned_labels)
        total_occurrences += len(labels)
        assigned_occurrences += len(assigned)
        trs_with_labels += int(bool(labels))
        trs_with_assigned += int(bool(assigned))

    print("\nAnnotation coverage")
    if total_occurrences:
        print(f"Label occurrence coverage: {100 * assigned_occurrences / total_occurrences:.1f}%")
    if trs_with_labels:
        print(f"TR coverage: {100 * trs_with_assigned / trs_with_labels:.1f}%")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
