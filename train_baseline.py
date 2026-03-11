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


def main():
    """
    Baseline YOLOv8 training on VOC for object detection.

    This script reproduces the baseline experiment described in the project:
    - YOLOv8n
    - VOC dataset
    - imgsz=640
    - epochs=50
    """
    # Load the official YOLOv8n pretrained weights
    model = YOLO("yolov8n.pt")

    repo_root = Path(__file__).resolve().parents[1]
    voc_root = (repo_root / "dataset" / "VOC").resolve()

    # Ultralytics expects `data` to be a YAML path (string). On Windows, using a
    # YAML with relative `path:` can be resolved against Ultralytics datasets_dir,
    # so we generate a small YAML with an absolute `path:` here.
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

    # Train on VOC with default augmentation and 640x640 input size
    model.train(
        task="detect",
        data=str(data_yaml),
        epochs=10,
        imgsz=416,
        batch=16,
        fraction=0.1,
        workers=0,
        project="runs_baseline",
        name="yolov8n_voc_baseline",
    )


if __name__ == "__main__":
    main()
