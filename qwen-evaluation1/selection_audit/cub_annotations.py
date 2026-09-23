"""Official CUB annotations and deterministic class-stratified train sampling."""
from collections import defaultdict
from pathlib import Path
import random


MACRO_PARTS = {
    "head": {"beak", "crown", "forehead", "left eye", "right eye", "nape", "throat"},
    "torso": {"back", "belly", "breast"},
    "wings": {"left wing", "right wing"},
    "legs": {"left leg", "right leg"},
    "tail": {"tail"},
}


class CUBAnnotations:
    def __init__(self, root):
        self.root = Path(root).resolve()
        def table(name):
            return {int(line.split()[0]): line.strip().split(maxsplit=1)[1]
                    for line in (self.root / name).read_text().splitlines() if line.strip()}
        paths = table("images.txt")
        labels = table("image_class_labels.txt")
        split = table("train_test_split.txt")
        names = table("parts/parts.txt")
        boxes = table("bounding_boxes.txt")
        assert paths.keys() == labels.keys() == split.keys() == boxes.keys()
        self.images = {i: {"image_id": i, "relative_path": "images/" + p,
                           "class_id": int(labels[i]), "is_train": int(split[i]),
                           "bbox": list(map(float, boxes[i].split())), "parts": []}
                       for i, p in paths.items()}
        self.by_path = {r["relative_path"]: i for i, r in self.images.items()}
        for line in (self.root / "parts/part_locs.txt").read_text().splitlines():
            image_id, part_id, x, y, visible = line.split()
            name = names[int(part_id)]
            macro = next(m for m, parts in MACRO_PARTS.items() if name in parts)
            self.images[int(image_id)]["parts"].append(
                {"part_id": int(part_id), "name": name, "macro": macro,
                 "x": float(x), "y": float(y), "visible": int(visible)})
        assert all(len(r["parts"]) == len(names) for r in self.images.values())

    def sample_train(self, seed=42, limit=None):
        groups = defaultdict(list)
        for image_id, item in sorted(self.images.items()):
            if item["is_train"] == 1:
                groups[item["class_id"]].append(image_id)
        rng = random.Random(seed)
        ids = [rng.choice(groups[c]) for c in sorted(groups)]
        # Shuffle classes before a smoke subset, avoiding the first 20 classes.
        rng.shuffle(ids)
        return ids if limit is None else ids[:limit]
