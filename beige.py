import os
import io
import warnings
import argparse
import logging
import json
import subprocess
from pathlib import Path
import numpy as np
import cv2
from PIL import Image


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
    logger = logging.getLogger("beige")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    if not quiet:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger


def verify_input_dir(input_dir):
    logger = logging.getLogger("beige")
    png_files = list(Path(input_dir).glob("*.png"))
    if len(png_files) < 300:
        raise Exception(
            f"Input directory must contain at least 300 PNG files. Found {len(png_files)}."
        )

    for i in range(300):
        if not (Path(input_dir) / f"{i}.png").exists():
            raise Exception(f"File {i}.png not found in input directory.")

    logger.info("Input directory verified successfully.")


def process_images(input_dir):
    logger = logging.getLogger("beige")
    proc_dir = Path(input_dir).parent / "proc"
    proc_dir.mkdir(exist_ok=True)

    for i in range(300):
        img_path = Path(input_dir) / f"{i}.png"
        img = cv2.imread(str(img_path))

        # Apply 3x3 median filter
        filtered_img = cv2.medianBlur(img, 3)

        # Convert to RGB for PIL
        rgb_img = cv2.cvtColor(filtered_img, cv2.COLOR_BGR2RGB)

        # Apply JPEG compression
        buffer = io.BytesIO()
        Image.fromarray(rgb_img).save(buffer, format="JPEG", quality=95)
        buffer.seek(0)
        compressed_img = Image.open(buffer)

        # Save as PNG
        output_path = proc_dir / f"{i}.png"
        compressed_img.save(str(output_path), format="PNG")

    logger.info(f"Processed images saved in {proc_dir}")
    return proc_dir


def run_decode_scripts(proc_dir, output_dir):
    logger = logging.getLogger("beige")

    # StegaStamp decoding
    stegastamp_images = [str(proc_dir / f"{i}.png") for i in range(150)]
    stegastamp_messages = sorted(
        Path("data/encoded/stegastamp/messages").glob("*.txt"),
        key=lambda x: int(x.stem),
    )
    subprocess.run(
        [
            "python",
            "decode/stegastamp/decode.py",
            "-p",
            *stegastamp_images,
            "-m",
            *stegastamp_messages,
            "-o",
            output_dir,
        ],
        check=True,
    )
    logger.info("StegaStamp decoding completed.")

    # Gaussian Shading decoding
    gaussianshading_images = [str(proc_dir / f"{i}.png") for i in range(150, 300)]
    gaussianshading_messages = sorted(
        Path("data/encoded/gaussianshading/messages").glob("*.pkl"),
        key=lambda x: int(x.stem),
    )
    subprocess.run(
        [
            "python",
            "decode/gaussianshading/decode.py",
            "-p",
            *gaussianshading_images,
            "-m",
            *gaussianshading_messages,
            "-o",
            output_dir,
        ],
        check=True,
    )
    logger.info("Gaussian Shading decoding completed.")


def run_metric_script(proc_dir, output_dir):
    logger = logging.getLogger("beige")

    subprocess.run(
        [
            "python",
            "metric/metric.py",
            "-i",
            str(proc_dir),
            "-r",
            "data/beige",
            "-o",
            output_dir,
        ],
        check=True,
    )

    logger.info("Metric calculation completed.")


def compute_performance(output_dir):
    STEGASTAMP_THRESHOLD = 0.35
    GAUSSIANSHADING_THRESHOLD = 0.4062460938

    # Load JSON files
    stegastamp_data = load_json(os.path.join(output_dir, "stegastamp-decode.json"))
    gaussianshading_data = load_json(
        os.path.join(output_dir, "gaussianshading-decode.json")
    )

    # Process StegaStamp data
    results = []
    for idx in range(150):
        distance = stegastamp_data[str(idx)]
        results.append(float(distance < STEGASTAMP_THRESHOLD))
    # Process Gaussian Shading data
    for idx in range(150, 300):
        distance = gaussianshading_data[str(idx)]
        results.append(float(distance < GAUSSIANSHADING_THRESHOLD))

    # Compute and return the mean
    return sum(results) / len(results)


def compute_quality(output_dir):
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
    return normalized_quality


def save_results(output_dir, performance, quality):
    results = {
        "performance": performance,
        "quality": quality,
        "score": float(np.sqrt(performance**2 + quality**2)),
    }
    save_json(results, os.path.join(output_dir, "scores.json"))


def main(input_dir, output_dir, quiet=False):
    logger = setup_logger(quiet)

    try:
        verify_input_dir(input_dir)
        proc_dir = process_images(input_dir)
        run_decode_scripts(proc_dir, output_dir)
        run_metric_script(proc_dir, output_dir)
        performance = compute_performance(output_dir)
        quality = compute_quality(output_dir)
        save_results(output_dir, performance, quality)

        logger.info("All operations completed successfully.")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Beige image processing script")
    parser.add_argument("-i", "--input_dir", required=True, help="Input directory path")
    parser.add_argument(
        "-o", "--output_dir", required=True, help="Output directory path"
    )
    args = parser.parse_args()

    main(args.input_dir, args.output_dir)
