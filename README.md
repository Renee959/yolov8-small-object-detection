## Small Object Detection Improvement Based on YOLOv8

This project implements and compares a **baseline YOLOv8 model** and an **improved model focused on small object detection** on the VOC dataset.

The improvements include:

- **Data augmentation**: stronger mosaic, mixup, and HSV augmentation.
- **Higher input resolution**: from 640 to 960 for better small-object features.
- **Loss optimization**: adjust classification/box weights and focal loss gamma.

The repository is structured to be easy to reproduce for a course project or thesis and to showcase on GitHub / resume.

---

## 1. Environment Setup

```bash
conda create -n yolov8 python=3.9
conda activate yolov8

pip install -r requirements.txt
```

Quick check:

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
print("YOLOv8 loaded successfully")
```

---

## 2. Dataset Preparation (VOC)

Download the **Pascal VOC 2007+2012** dataset and unpack it so that the images and labels follow the standard YOLO layout.

The default configuration in `dataset/voc.yaml` assumes the following structure (you can change it if needed):

```text
dataset/
  VOC/
    images/
      train/
      val/
    labels/
      train/
      val/
```

- `dataset/voc.yaml` describes the dataset and class names.
- If your paths are different, edit `path`, `train`, and `val` fields in `dataset/voc.yaml`.

You can also start with the built-in VOC configuration in Ultralytics by changing `data="dataset/voc.yaml"` to `data="VOC"` in the training scripts.

### 2.1 One-command VOC2007+2012 → YOLO (Windows/Linux)

This repo includes a helper script that:

- Downloads VOC2007 trainval + test and VOC2012 trainval
- Extracts to `dataset/raw/VOCdevkit/`
- Converts VOC XML annotations to YOLO txt labels
- Merges VOC2007+2012 trainval into `train`
- Uses VOC2007 `test` as `val` (common practice)

Run from the repo root:

```bash
python tools/prepare_voc.py --download
```

Outputs:

```text
dataset/
  raw/
    VOCdevkit/
      VOC2007/ ...
      VOC2012/ ...
  VOC/
    images/
      train/
      val/
    labels/
      train/
      val/
```

If you prefer VOC2007 `val` (instead of `test`) as validation:

```bash
python tools/prepare_voc.py --download --val_from VOC2007_val
```

---

## 3. Baseline Training

Baseline YOLOv8n on VOC:

```bash
cd training
python train_baseline.py
```

Key settings:

- `model="yolov8n.pt"`
- `data="dataset/voc.yaml"`
- `epochs=50`
- `imgsz=640`
- `batch=16`

Outputs (by default):

- `runs_baseline/yolov8n_voc_baseline/`
  - `results.png`
  - `confusion_matrix.png`
  - `weights/best.pt`

Record metrics such as:

- `mAP50`
- `mAP50-95`
- `Precision`
- `Recall`

These form the **baseline** row of your comparison table.

---

## 4. Improved Model for Small Objects

Run the improved training script:

```bash
cd training
python train_augmented.py
```

This applies three improvements:

1. **Data augmentation**
   - `mosaic=1.0`
   - `mixup=0.2`
   - `hsv_h=0.015`, `hsv_s=0.7`, `hsv_v=0.4`

2. **Higher input resolution**
   - `imgsz=960` (improves representation of small objects)

3. **Loss adjustments**
   - `cls=0.7`, `box=7.5`, `dfl=1.5`
   - `fl_gamma=2.0` (focal loss gamma to emphasize hard examples / small objects)

Outputs:

- `runs_improved/yolov8n_voc_small_objects/`
  - `results.png`
  - `confusion_matrix.png`
  - `weights/best.pt`

Use these to fill the **improved** rows in your result table.

---

## 5. Result Comparison

Create a simple table in your report or in `results/` (e.g. `results/map_comparison.md` or `.png`):

| Model                                  | mAP50 | mAP50-95 | Recall | Precision |
| -------------------------------------- | ----- | -------- | ------ | --------- |
| Baseline YOLOv8n (640)                |       |          |        |           |
| + Data Augmentation + High Resolution |       |          |        |           |
| + Loss Optimization (Small Objects)   |       |          |        |           |

Fill in the values by reading `results.png` and the training logs from each run.

You can also save qualitative examples (good/bad cases) into `results/`:

- `results/detection_examples.png`
- `results/small_objects_closeup.png`

---

## 6. Inference (Image / Video)

Use the generic inference script:

```bash
python inference/detect.py \
  --weights runs_improved/yolov8n_voc_small_objects/weights/best.pt \
  --source path/to/test.jpg \
  --save_dir results/inference \
  --show
```

- For a **directory of images**, set `--source path/to/dir/`.
- For **video**, pass a video file: `--source path/to/video.mp4`.

Annotated outputs are saved under `results/inference/predictions/`.

---

## 7. Demo Video / GIF

To create a demo video for your GitHub or report:

1. Prepare an input video at `demo/input_video.mp4` (or any path you like).
2. Run:

```bash
python demo/make_demo_video.py \
  --weights runs_improved/yolov8n_voc_small_objects/weights/best.pt \
  --source demo/input_video.mp4 \
  --output_dir demo
```

The output demo video will be in `demo/demo_video/` (e.g. `demo/demo_video/video.mp4`).
You can convert this to a GIF using tools like ffmpeg or an online converter.

---

## 8. Suggested GitHub Layout

When you push to GitHub, the repository will roughly look like:

```text
.
├── dataset/
│   ├── VOC/               # your VOC data (not necessarily committed)
│   └── voc.yaml
│
├── training/
│   ├── train_baseline.py
│   └── train_augmented.py
│
├── inference/
│   └── detect.py
│
├── results/
│   ├── README.md
│   ├── map_comparison.png        # to be generated by you
│   └── detection_examples.png    # to be generated by you
│
├── demo/
│   ├── input_video.mp4           # your test video (optional)
│   └── make_demo_video.py
│
├── requirements.txt
└── README.md
```

You can further add:

- Jupyter notebooks for analysis.
- A report in PDF / Markdown.
- Links to your demo video / GIF in the README.


