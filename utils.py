import os
import io
import sys
import shutil
import warnings
import argparse
import logging
import json
import subprocess
import multiprocessing
from tqdm.auto import tqdm
from pathlib import Path
import numpy as np
import cv2
from PIL import Image


THRESHOLD_FUNCTIONS = {
    "gaussianshading": [lambda x: x < 0.4062460938, lambda x: x < 0.402],
    "jigmark": [lambda x: x > 3.27, lambda x: x > 3.3],
    "prc": [lambda x: bool(int(x))],
    "stablesig": [lambda x: x < 0.29167, lambda x: x < 0.28],
    "stegastamp": [lambda x: x < 0.35, lambda x: x < 0.34],
    "trufo": [lambda x: x < 0.3167],
}
QUALITY_COEFFICIENTS = {
    "legacy_fid": 0.0015295359209778563,
    "clip_fid": 0.0050651488980776324,
    "psnr": -0.0022186489180419534,
    "ssim": -0.11337077856710862,
    "nmi": -0.09878221979274945,
    "lpips": 0.3412626374646173,
    "aesthetics": 0.044982716146790995,
    "artifacts": -0.1437622439320745,
}
QUALITY_OFFSETS = {
    "legacy_fid": 0.6282276268229566,
    "clip_fid": 0.22445010611941865,
    "psnr": 43.54757854447622,
    "ssim": 0.984229018845295,
    "nmi": 1.7536553655336136,
    "lpips": 0.014247652621287854,
    "aesthetics": -0.005184057521820067,
    "artifacts": -0.0015724570911377733,
}
# fmt: off
SHUFFLE_DICT = {'gaussianshading+jigmark': [208, 188, 12, 221, 239, 136, 230, 206, 52, 108, 290, 15, 184, 248, 22, 74, 270, 90, 229, 164, 190, 103, 124, 129, 222, 116, 194, 286, 234, 295, 92, 66, 8, 122, 219, 150, 176, 182, 299, 5, 223, 89, 81, 34, 55, 139, 236, 64, 7, 45], 'jigmark': [73, 213, 173, 106, 59, 253, 168, 26, 284, 153, 134, 145, 63, 293, 285, 224, 252, 111, 20, 46, 156, 228, 273, 27, 144, 259, 37, 97, 191, 135, 118, 160, 264, 214, 238, 76, 212, 225, 255, 237, 282, 44, 272, 189, 152, 158, 101, 54, 181, 18], 'prc': [281, 250, 126, 171, 71, 227, 245, 205, 288, 215, 154, 159, 33, 83, 249, 60, 167, 280, 110, 21, 29, 146, 16, 56, 75, 109, 175, 201, 161, 4, 96, 166, 61, 67, 137, 198, 262, 279, 40, 268, 13, 107, 220, 3, 157, 125, 24, 30, 77, 291], 'stablesig': [210, 19, 254, 241, 266, 80, 51, 2, 235, 104, 179, 86, 10, 199, 58, 41, 14, 155, 50, 292, 233, 123, 200, 62, 187, 226, 130, 209, 260, 43, 114, 138, 294, 218, 149, 112, 247, 98, 217, 93, 216, 162, 36, 178, 113, 0, 94, 275, 95, 296], 'stablesig+stegastamp': [261, 240, 69, 49, 48, 85, 297, 141, 207, 23, 246, 148, 143, 78, 180, 100, 204, 131, 267, 298, 196, 6, 68, 203, 84, 170, 121, 140, 256, 274, 142, 257, 91, 82, 283, 11, 119, 102, 35, 57, 169, 231, 65, 1, 120, 269, 186, 42, 105, 132], 'trufo': [79, 17, 263, 232, 38, 133, 53, 258, 128, 28, 183, 163, 151, 244, 202, 31, 32, 127, 185, 278, 271, 147, 276, 177, 99, 197, 243, 115, 265, 72, 25, 165, 287, 174, 289, 39, 193, 88, 70, 87, 242, 277, 211, 9, 195, 251, 192, 117, 47, 172]}
# fmt: on


def parse_args():
    parser = argparse.ArgumentParser(description="Beige box evaluation script")
    parser.add_argument("-i", "--input_dir", required=True, help="Input directory path")
    parser.add_argument(
        "-o", "--output_dir", required=True, help="Output directory path"
    )
    args = parser.parse_args()
    return args


def sorted_listdir(dir, ext):
    return sorted(
        Path(dir).glob(f"*{ext}"),
        key=lambda x: int(x.stem),
    )


def load_json(filepath):
    try:
        with open(filepath, "r") as json_file:
            return json.load(json_file)
    except json.JSONDecodeError:
        warnings.warn(f"Found invalid JSON file {filepath}, deleting")
        os.remove(filepath)
        return None


def save_json(data, filepath):
    with open(filepath, "w") as json_file:
        json.dump(data, json_file)


def setup_logger(quiet=False):
    logger = logging.getLogger("eval")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "[%(asctime)s | %(name)s | %(levelname)s] %(message)s"
    )

    if not quiet:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger


import logging
from pathlib import Path
import shutil

def verify_input_dir(input_dir):
    logger = logging.getLogger("eval")
    input_path = Path(input_dir)

    # Check if there's a single subfolder
    subfolders = [f for f in input_path.iterdir() if f.is_dir()]
    if len(subfolders) == 1:
        subfolder = subfolders[0]
        logger.info(f"Found a single subfolder: {subfolder.name}")
        png_files = list(subfolder.glob("*.png"))
    else:
        png_files = list(input_path.glob("*.png"))

    if len(png_files) < 300:
        raise Exception(
            f"Input directory must contain at least 300 PNG files. Found {len(png_files)}."
        )

    # Check for required files
    missing_files = []
    for i in range(300):
        file_name = f"{i}.png"
        if not any(file.name == file_name for file in png_files):
            missing_files.append(file_name)

    if missing_files:
        raise Exception(f"The following required files are missing: {', '.join(missing_files)}")

    # If files are in a subfolder, move them to input_dir
    if len(subfolders) == 1:
        logger.info("Moving PNG files from subfolder to input directory.")
        for png_file in png_files:
            shutil.move(str(png_file), str(input_path / png_file.name))
        subfolder.rmdir()
        logger.info(f"Removed subfolder: {subfolder.name}")

    logger.info("Input directory verified successfully.")


def process_single_image(args):
    i, input_dir, proc_dir = args
    img_path = Path(input_dir) / f"{i}.png"
    img = cv2.imread(str(img_path))

    # Apply 3x3 median filter
    filtered_img = cv2.medianBlur(img, 3)

    # Convert to RGB for PIL
    rgb_img = cv2.cvtColor(filtered_img, cv2.COLOR_BGR2RGB)

    # Apply JPEG compression
    buffer = io.BytesIO()
    Image.fromarray(rgb_img).save(buffer, format="JPEG", quality=100)
    buffer.seek(0)
    compressed_img = Image.open(buffer)

    # Save as PNG
    output_path = proc_dir / f"{i}.png"
    compressed_img.save(str(output_path), format="PNG")


def process_images(mode, input_dir):
    logger = logging.getLogger("eval")

    proc_dir = Path(input_dir).parent / "proc"
    proc_dir.mkdir(exist_ok=True)

    # Detect number of CPUs
    num_cpus = multiprocessing.cpu_count()

    # Create a pool of workers
    with multiprocessing.Pool(processes=num_cpus) as pool:
        # Prepare arguments for each task
        args = [(i, input_dir, proc_dir) for i in range(300)]

        # Use tqdm to show progress
        list(
            tqdm(
                pool.imap(process_single_image, args),
                total=300,
                desc="Processing images",
            )
        )

    # Weak watermark excempt from processing
    if mode == "black":
        for idx in SHUFFLE_DICT["stablesig"]:
            shutil.move(
                os.path.join(input_dir, f"{idx}.png"),
                os.path.join(proc_dir, f"{idx}.png"),
            )

    logger.info(f"Processed images saved in {proc_dir}")
    return proc_dir


def run_script(mode, image_paths, message_paths, output_dir):
    logger = logging.getLogger("eval")

    assert len(image_paths) == len(
        message_paths
    ), f"Number of images and messages must be equal. Found {len(image_paths)} images and {len(message_paths)} messages."
    subprocess.run(
        [
            sys.executable,
            f"decode/{mode}/decode.py",
            "-p",
            *image_paths,
            "-m",
            *message_paths,
            "-o",
            output_dir,
        ],
        check=True,
    )

    logger.info(f"Watermark *** decoding completed.")


def metric(mode, proc_dir, output_dir):
    logger = logging.getLogger("eval")

    assert mode in ["beige", "black"]
    subprocess.run(
        [
            "python",
            "metric/metric.py",
            "-i",
            str(proc_dir),
            "-r",
            f"data/{mode}",
            "-o",
            output_dir,
        ],
        check=True,
    )

    logger.info("Metric calculation completed.")


def compute_quality(output_dir):
    logger = logging.getLogger("eval")

    # Load JSON files
    attacked_metric = load_json(os.path.join(output_dir, "metric.json"))
    clean_metric = load_json("data/metric/beige-metric.json")

    # Process metrics
    results = {key: 0.0 for key in QUALITY_COEFFICIENTS.keys()}
    for idx in range(300):
        for key in results.keys():
            if key not in ["aesthetics", "artifacts"]:
                results[key] += attacked_metric[str(idx)][key]
            else:
                results[key] += (
                    clean_metric[str(idx)][key] - attacked_metric[str(idx)][key]
                )

    # Normalize and return the quality
    normalized_quality = 0.0
    for key, value in results.items():
        normalized_quality += (
            value / 300 - QUALITY_OFFSETS[key]
        ) * QUALITY_COEFFICIENTS[key]

    logger.info("Quality calculation completed.")
    return normalized_quality - 0.1


def save_results(output_dir, performance, quality):
    logger = logging.getLogger("eval")

    results = {
        "performance": performance,
        "quality": quality,
        "score": float(np.sqrt(performance**2 + quality**2)),
    }
    save_json(results, os.path.join(output_dir, "scores.json"))

    logger.info("Results saved successfully.")
