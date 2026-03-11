import argparse
import os
import shutil
import tarfile
import urllib.request
import xml.etree.ElementTree as ET


VOC_CLASSES = [
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

VOC_URLS = {
    "VOC2007_trainval": (
        "https://thor.robots.ox.ac.uk/pascal/VOC/voc2007/"
        "VOCtrainval_06-Nov-2007.tar"
    ),
    "VOC2007_test": (
        "https://thor.robots.ox.ac.uk/pascal/VOC/voc2007/"
        "VOCtest_06-Nov-2007.tar"
    ),
    "VOC2012_trainval": (
        "https://thor.robots.ox.ac.uk/pascal/VOC/voc2012/"
        "VOCtrainval_11-May-2012.tar"
    ),
}


def _mkdir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def download(url: str, dst_path: str) -> None:
    _mkdir(os.path.dirname(dst_path))
    if os.path.exists(dst_path) and os.path.getsize(dst_path) > 0:
        return
    print(f"Downloading: {url}")
    urllib.request.urlretrieve(url, dst_path)


def extract_tar(tar_path: str, dst_dir: str) -> None:
    print(f"Extracting: {tar_path}")
    _mkdir(dst_dir)
    with tarfile.open(tar_path, "r:*") as tf:
        tf.extractall(dst_dir)


def read_ids(imageset_txt: str) -> list[str]:
    with open(imageset_txt, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines() if line.strip()]


def voc_xml_to_yolo(xml_path: str) -> tuple[list[str], str]:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    if size is None:
        raise ValueError(f"Missing <size> in {xml_path}")
    w = float(size.findtext("width"))
    h = float(size.findtext("height"))
    if w <= 0 or h <= 0:
        raise ValueError(f"Invalid image size in {xml_path}")

    lines: list[str] = []
    for obj in root.findall("object"):
        cls = obj.findtext("name")
        if not cls or cls not in VOC_CLASSES:
            continue
        difficult = obj.findtext("difficult")
        if difficult is not None and difficult.strip() == "1":
            # Keep consistent with common VOC->YOLO conversions: skip difficult
            continue
        bnd = obj.find("bndbox")
        if bnd is None:
            continue

        xmin = float(bnd.findtext("xmin"))
        ymin = float(bnd.findtext("ymin"))
        xmax = float(bnd.findtext("xmax"))
        ymax = float(bnd.findtext("ymax"))

        # clamp
        xmin = max(0.0, min(xmin, w - 1))
        xmax = max(0.0, min(xmax, w - 1))
        ymin = max(0.0, min(ymin, h - 1))
        ymax = max(0.0, min(ymax, h - 1))
        if xmax <= xmin or ymax <= ymin:
            continue

        x = ((xmin + xmax) / 2.0) / w
        y = ((ymin + ymax) / 2.0) / h
        bw = (xmax - xmin) / w
        bh = (ymax - ymin) / h

        cls_id = VOC_CLASSES.index(cls)
        lines.append(f"{cls_id} {x:.6f} {y:.6f} {bw:.6f} {bh:.6f}")

    return lines, root.findtext("filename") or ""


def copy_image(src_jpg: str, dst_jpg: str) -> None:
    _mkdir(os.path.dirname(dst_jpg))
    if os.path.exists(dst_jpg):
        return
    shutil.copy2(src_jpg, dst_jpg)


def write_label(lines: list[str], dst_txt: str) -> None:
    _mkdir(os.path.dirname(dst_txt))
    with open(dst_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        if lines:
            f.write("\n")


def convert_split(
    voc_dir: str, ids: list[str], split: str, out_dir: str, prefix: str
) -> int:
    img_dir = os.path.join(voc_dir, "JPEGImages")
    ann_dir = os.path.join(voc_dir, "Annotations")

    out_img_dir = os.path.join(out_dir, "images", split)
    out_lbl_dir = os.path.join(out_dir, "labels", split)
    _mkdir(out_img_dir)
    _mkdir(out_lbl_dir)

    n = 0
    for image_id in ids:
        xml_path = os.path.join(ann_dir, f"{image_id}.xml")
        jpg_src = os.path.join(img_dir, f"{image_id}.jpg")
        if not os.path.exists(xml_path) or not os.path.exists(jpg_src):
            continue

        yolo_lines, _ = voc_xml_to_yolo(xml_path)
        # Keep empty label files too (valid for YOLO training)
        out_name = f"{prefix}_{image_id}"
        jpg_dst = os.path.join(out_img_dir, f"{out_name}.jpg")
        txt_dst = os.path.join(out_lbl_dir, f"{out_name}.txt")

        copy_image(jpg_src, jpg_dst)
        write_label(yolo_lines, txt_dst)
        n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Download/merge VOC2007+2012 and convert to YOLO format."
    )
    ap.add_argument(
        "--raw_dir",
        default=os.path.join("dataset", "raw"),
        help="Where to download/extract VOCdevkit",
    )
    ap.add_argument(
        "--out_dir",
        default=os.path.join("dataset", "VOC"),
        help="Output YOLO dataset directory",
    )
    ap.add_argument(
        "--download",
        action="store_true",
        help="Download VOC2007 trainval+test and VOC2012 trainval tars into raw_dir",
    )
    ap.add_argument(
        "--val_from",
        choices=["VOC2007_test", "VOC2007_val"],
        default="VOC2007_test",
        help=(
            "Validation split source: VOC2007 test (recommended) or VOC2007 val."
        ),
    )
    args = ap.parse_args()

    raw_dir = os.path.abspath(args.raw_dir)
    out_dir = os.path.abspath(args.out_dir)
    _mkdir(raw_dir)
    _mkdir(out_dir)

    if args.download:
        for name, url in VOC_URLS.items():
            tar_path = os.path.join(raw_dir, f"{name}.tar")
            download(url, tar_path)
            extract_tar(tar_path, raw_dir)

    vocdevkit = os.path.join(raw_dir, "VOCdevkit")
    voc2007 = os.path.join(vocdevkit, "VOC2007")
    voc2012 = os.path.join(vocdevkit, "VOC2012")

    for p in [voc2007, voc2012]:
        if not os.path.isdir(p):
            raise SystemExit(
                f"Missing {p}. Run with --download or extract VOCdevkit into "
                f"{raw_dir} first."
            )

    train07 = read_ids(
        os.path.join(voc2007, "ImageSets", "Main", "trainval.txt")
    )
    train12 = read_ids(
        os.path.join(voc2012, "ImageSets", "Main", "trainval.txt")
    )
    if args.val_from == "VOC2007_test":
        val_ids = read_ids(
            os.path.join(voc2007, "ImageSets", "Main", "test.txt")
        )
    else:
        val_ids = read_ids(
            os.path.join(voc2007, "ImageSets", "Main", "val.txt")
        )

    print("Converting train split (VOC2007 trainval + VOC2012 trainval)...")
    n_train = 0
    n_train += convert_split(voc2007, train07, "train", out_dir, prefix="07")
    n_train += convert_split(voc2012, train12, "train", out_dir, prefix="12")

    print(f"Converting val split ({args.val_from})...")
    n_val = convert_split(voc2007, val_ids, "val", out_dir, prefix="07")

    print("Done.")
    print(f"Output: {out_dir}")
    print(f"Train images: {n_train}, Val images: {n_val}")
    print("Expected ultralytics YAML:")
    print("  path: ./dataset/VOC")
    print("  train: images/train")
    print("  val: images/val")


if __name__ == "__main__":
    main()
