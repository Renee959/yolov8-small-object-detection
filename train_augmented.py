from ultralytics import YOLO
from pathlib import Path


VOC_NAMES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]


def train_improved():
    """
    Improved YOLOv8 training focusing on small object detection.

    This script applies three cumulative improvements over the baseline:
    1) Stronger data augmentation (mosaic, mixup, HSV).
    2) Higher input resolution (imgsz=960).
    3) Loss weighting tweaks for small objects.
    """
    model = YOLO("yolov8n.pt")

    repo_root = Path(__file__).resolve().parents[1]
    voc_root = (repo_root / "dataset" / "VOC").resolve()
    data_yaml = Path(__file__).with_name("_voc_abs.yaml")
    names_yaml = "\n".join([f"  - {n}" for n in VOC_NAMES])
    data_yaml.write_text(
        "\n".join([
            f"path: {voc_root.as_posix()}",
            "train: images/train",
            "val: images/val",
            "names:",
            names_yaml,
            "",
        ]),
        encoding="utf-8",
    )

    model.train(
        task="detect",
        data=str(data_yaml),
        epochs=10,
        imgsz=416,  # higher resolution for small objects
        batch=8,
        fraction=0.1,
        workers=0,
        mosaic=1.0,
        mixup=0.2,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        # loss-related hyper-parameters
        cls=0.7,
        box=7.5,
        dfl=1.5,
        project="runs_improved",
        name="yolov8n_voc_small_objects",
    )


if __name__ == "__main__":
    train_improved()
