import os
import torch
from huggingface_hub import constants

current_dir = os.getcwd()
torch.hub.set_dir(os.path.join(current_dir, "cache", "pytorch"))
os.environ["TRANSFORMERS_CACHE"] = os.path.join(current_dir, "cache", "huggingface")
os.environ["HF_HOME"] = os.path.join(current_dir, "cache", "huggingface")
constants.HUGGINGFACE_HUB_CACHE = os.path.join(current_dir, "cache", "huggingface")

import argparse
import warnings
import json
from tqdm.auto import tqdm
from transformers import logging
import torch.multiprocessing as mp
from PIL import Image
from modules import (
    compute_fid,
    compute_image_distance_repeated,
    load_perceptual_models,
    compute_perceptual_metric_repeated,
    load_aesthetics_and_artifacts_models,
    compute_aesthetics_and_artifacts_scores,
    load_open_clip_model_preprocess_and_tokenizer,
    compute_clip_score,
)

warnings.filterwarnings("ignore")
logging.set_verbosity_error()

QUALITY_METRICS = {
    "legacy_fid": "Legacy FID",
    "clip_fid": "CLIP FID",
    "psnr": "PSNR",
    "ssim": "SSIM",
    "nmi": "Normed Mutual-Info",
    "lpips": "LPIPS",
    "aesthetics": "Delta Aesthetics",
    "artifacts": "Delta Artifacts",
}
DELTA_METRICS = ["aesthetics", "artifacts"]
PARALLEL_METRICS = []


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


def process_single(mode, indices, path, clean_path, attacked_path, quiet, limit):
    if mode.endswith("_fid"):
        num_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            raise RuntimeError("No GPUs available for processing")
        if not quiet:
            print(f"Using {num_gpus} GPUs for processing")
        metric = float(
            compute_fid(
                clean_path,
                attacked_path,
                mode=mode.split("_")[0],
                device=torch.device("cuda"),
                batch_size=64,
                num_workers=8,
                verbose=not quiet,
            )
        )
        return {idx: metric for idx in indices}

    elif mode in ["psnr", "ssim", "nmi"]:
        if not quiet:
            print(f"Using {os.cpu_count()} CPUs for processing")
        clean_images = [
            Image.open(os.path.join(clean_path, f"{idx}.png")) for idx in indices
        ]
        attacked_images = [
            Image.open(os.path.join(attacked_path, f"{idx}.png")) for idx in indices
        ]
        metrics = compute_image_distance_repeated(
            clean_images,
            attacked_images,
            metric_name=mode,
            num_workers=8,
            verbose=not quiet,
        )
        results = {idx: metric for idx, metric in zip(indices, metrics)}
        [image.close() for image in clean_images]
        [image.close() for image in attacked_images]
        return results

    elif mode in ["lpips", "watson"]:
        if not quiet:
            print(f"Using 1 GPU and {os.cpu_count()} CPUs for processing")
        clean_images = [
            Image.open(os.path.join(clean_path, f"{idx}.png")) for idx in indices
        ]
        attacked_images = [
            Image.open(os.path.join(attacked_path, f"{idx}.png")) for idx in indices
        ]
        model = load_perceptual_models(
            mode,
            mode="alex" if mode == "lpips" else "dft",
            device=torch.device("cuda"),
        )

        batch_size = 32
        pbar = tqdm(
            total=len(indices), desc=f"Computing {mode} metrics on images", unit="image"
        )
        metrics = []
        for it in range(0, len(indices), batch_size):
            metrics.extend(
                compute_perceptual_metric_repeated(
                    clean_images[it : min(it + batch_size, len(indices))],
                    attacked_images[it : min(it + batch_size, len(indices))],
                    metric_name=mode,
                    mode="alex" if mode == "lpips" else "dft",
                    model=model,
                    device=torch.device("cuda"),
                )
            )
            pbar.update(min(batch_size, len(indices) - it))
        pbar.close()
        results = {idx: metric for idx, metric in zip(indices, metrics)}
        [image.close() for image in clean_images]
        [image.close() for image in attacked_images]
        return results

    elif mode == "aesthetics_and_artifacts":
        if not quiet:
            print(f"Using 1 GPU and {os.cpu_count()} CPUs for processing")
        model = load_aesthetics_and_artifacts_models(device=torch.device("cuda"))
        images = [Image.open(os.path.join(path, f"{idx}.png")) for idx in indices]

        batch_size = 8
        pbar = tqdm(
            total=len(indices), desc=f"Computing {mode} metrics on images", unit="image"
        )
        metrics = []
        for it in range(0, len(indices), batch_size):
            aesthetics, artifacts = compute_aesthetics_and_artifacts_scores(
                images[it : min(it + batch_size, len(indices))],
                model,
                device=torch.device("cuda"),
            )
            metrics.extend(list(zip(aesthetics, artifacts)))
            pbar.update(min(batch_size, len(indices) - it))
        pbar.close()
        results = {idx: metric for idx, metric in zip(indices, metrics)}
        [image.close() for image in images]
        return results


def init_model(mode, gpu):
    if mode == "aesthetics_and_artifacts":
        return load_aesthetics_and_artifacts_models(device=torch.device(f"cuda:{gpu}"))
    elif mode == "clip_score":
        return load_open_clip_model_preprocess_and_tokenizer(
            device=torch.device(f"cuda:{gpu}")
        )


def load_files(mode, path, clean_path, attacked_path, indices):
    if mode == "aesthetics_and_artifacts":
        return [Image.open(os.path.join(path, f"{idx}.png")) for idx in indices]


def measure(mode, model, gpu, inputs):
    if mode == "aesthetics_and_artifacts":
        aesthetics, artifacts = compute_aesthetics_and_artifacts_scores(
            inputs, model, device=torch.device(f"cuda:{gpu}")
        )
        return list(zip(aesthetics, artifacts))
    elif mode == "clip_score":
        return compute_clip_score(
            inputs[0], inputs[1], model, device=torch.device(f"cuda:{gpu}")
        )


def worker(mode, gpu, path, clean_path, attacked_path, indices, lock, counter, results):
    model = init_model(mode, gpu)
    batch_size = {
        "aesthetics_and_artifacts": 8,
        "clip_score": 8,
    }[mode]
    for it in range(0, len(indices), batch_size):
        inputs = load_files(
            mode,
            path,
            clean_path,
            attacked_path,
            indices[it : min(it + batch_size, len(indices))],
        )
        metrics = measure(mode, model, gpu, inputs)
        with lock:
            counter.value += min(batch_size, len(indices) - it)
            for idx, metric in zip(
                indices[it : min(it + batch_size, len(indices))], metrics
            ):
                results[idx] = metric


def process_parallel(mode, indices, path, clean_path, attacked_path, quiet):
    mp.set_start_method("spawn", force=True)  # Set start method to 'spawn'
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No GPUs available for processing")
    if not quiet:
        print(f"Using {num_gpus} GPUs for processing")

    num_workers = {
        "aesthetics_and_artifacts": num_gpus,
        "clip_score": num_gpus,
    }[mode]
    chunk_size = len(indices) // num_workers
    with mp.Manager() as manager:
        counter = manager.Value("i", 0)
        lock = manager.Lock()
        results = manager.dict()

        processes = []
        for rank in range(num_workers):
            start_idx = rank * chunk_size
            end_idx = None if rank == num_workers - 1 else (rank + 1) * chunk_size
            p = mp.Process(
                target=worker,
                args=(
                    mode,
                    rank % num_gpus,
                    path,
                    clean_path,
                    attacked_path,
                    indices[start_idx:end_idx],
                    lock,
                    counter,
                    results,
                ),
            )
            p.start()
            processes.append(p)

        with tqdm(
            total=len(indices), desc=f"Computing {mode} metrics on images", unit="image"
        ) as pbar:
            while True:
                with lock:
                    pbar.n = counter.value
                    pbar.refresh()
                    if counter.value >= len(indices):
                        break

        for p in processes:
            p.join()

        return dict(results)


def report(mode, output_path, results, quiet, limit):
    json_path = os.path.join(output_path, "metric.json")
    if (not os.path.exists(json_path)) or (data := load_json(json_path)) is None:
        data = {}
        for idx in range(limit):
            data[str(idx)] = {
                _mode: results.get(idx) if mode == _mode else None
                for _mode in QUALITY_METRICS.keys()
            }
    else:
        for idx, metric in results.items():
            data[str(idx)][mode] = metric
    save_json(data, json_path)
    if not quiet:
        print(f"Computed {mode} metrics saved to {json_path}")


def single_mode(mode, path, clean_path, attacked_path, output_path, quiet, limit):
    if not quiet:
        print(f"Computing {mode} metrics")
    indices = range(limit)

    if mode not in PARALLEL_METRICS:
        results = process_single(
            mode, indices, path, clean_path, attacked_path, quiet, limit
        )
    else:
        results = process_parallel(
            mode, indices, path, clean_path, attacked_path, quiet
        )
    if mode == "aesthetics_and_artifacts":
        report(
            "aesthetics",
            output_path,
            {idx: result[0] for idx, result in results.items()},
            quiet,
            limit,
        )
        report(
            "artifacts",
            output_path,
            {idx: result[1] for idx, result in results.items()},
            quiet,
            limit,
        )
    else:
        report(mode, output_path, results, quiet, limit)


def main(input_path, ref_path, output_path, quiet=False, limit=300):
    if not quiet:
        print(f"Computing all image quality metrics, {list(QUALITY_METRICS.keys())}")
    modes = QUALITY_METRICS.keys()

    if "aesthetics" in modes or "artifacts" in modes:
        modes = [mode for mode in modes if mode not in ["aesthetics", "artifacts"]] + [
            "aesthetics_and_artifacts"
        ]

    for mode in modes:
        single_mode(
            mode,
            path=input_path,
            clean_path=ref_path,
            attacked_path=input_path,
            output_path=output_path,
            quiet=quiet,
            limit=limit,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute image quality metrics.")
    parser.add_argument(
        "-i",
        "--input",
        dest="input_path",
        required=True,
        help="Path to the input images",
    )
    parser.add_argument(
        "-r",
        "--ref",
        dest="ref_path",
        required=True,
        help="Path to the reference images",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output_path",
        required=True,
        help="Path for the output directory",
    )
    args = parser.parse_args()

    main(
        input_path=args.input_path,
        ref_path=args.ref_path,
        output_path=args.output_path,
    )
