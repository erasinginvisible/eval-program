import os
import torch

current_dir = os.getcwd()
torch.hub.set_dir(os.path.join(current_dir, "cache", "pytorch"))

import warnings
import argparse
import json
from tqdm.auto import tqdm
from PIL import Image, ImageOps
import numpy as np
import onnxruntime as ort
import torch.multiprocessing as mp

warnings.filterwarnings("ignore")


def save_json(data, filepath):
    with open(filepath, "w") as json_file:
        json.dump(data, json_file)


def init_model(gpu):
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 1
    session_options.inter_op_num_threads = 1
    session_options.log_severity_level = 3
    return ort.InferenceSession(
        "model/stegastamp.onnx",
        providers=["CUDAExecutionProvider"],
        provider_options=[{"device_id": str(gpu)}],
        sess_options=session_options,
    )


def load_files(picture_paths):
    return np.stack(
        [
            np.array(
                ImageOps.fit(Image.open(path), (400, 400)),
                dtype=np.float32,
            )
            / 255.0
            for path in picture_paths
        ],
        axis=0,
    )


def decode(model, inputs):
    outputs = model.run(
        None,
        {
            "image": inputs,
            "secret": np.zeros((inputs.shape[0], 100), dtype=np.float32),
        },
    )
    return outputs[2].astype(bool)


def worker(gpu, picture_paths, lock, counter, results):
    model = init_model(gpu)
    batch_size = 4
    for it in range(0, len(picture_paths), batch_size):
        inputs = load_files(
            picture_paths[it : min(it + batch_size, len(picture_paths))]
        )
        messages = decode(model, inputs)
        with lock:
            counter.value += inputs.shape[0]
            for path, message in zip(
                picture_paths[it : min(it + batch_size, len(picture_paths))], messages
            ):
                results[int(path.split("/")[-1].split(".")[0])] = message


def process(picture_paths, quiet):
    mp.set_start_method("spawn", force=True)  # Set start method to 'spawn'
    # num_gpus = torch.cuda.device_count()
    num_gpus = 1
    if num_gpus == 0:
        raise RuntimeError("No GPUs available for processing")
    if not quiet:
        print(f"Using {num_gpus} GPUs for processing")

    num_workers = num_gpus
    chunk_size = len(picture_paths) // num_workers
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
                    rank % num_gpus,
                    picture_paths[start_idx:end_idx],
                    lock,
                    counter,
                    results,
                ),
            )
            p.start()
            processes.append(p)

        with tqdm(
            total=len(picture_paths), desc="Decoding images", unit="file", mininterval=5
        ) as pbar:
            while True:
                with lock:
                    pbar.n = counter.value
                    # pbar.refresh()
                    if counter.value >= len(picture_paths):
                        break

        for p in processes:
            p.join()

        return dict(results)


def load_message(message_path):
    with open(message_path, "r") as file:
        binary_string = file.read().strip()
    binary_list = [int(bit) for bit in binary_string]
    binary_array = np.array(binary_list, dtype=np.bool_)
    return binary_array


def bit_error_rate(pred, target):
    if not pred.dtype == target.dtype == bool:
        raise ValueError(f"Cannot compute BER for {pred.dtype} and {target.dtype}")
    return np.mean(pred != target)


def report(output_path, decoded_messages, picture_paths, message_paths, quiet):
    json_path = os.path.join(output_path, "e-decode.json")
    data = {}
    for picture_path, message_path in zip(picture_paths, message_paths):
        idx = int(picture_path.split("/")[-1].split(".")[0])
        gt_message = load_message(message_path)
        distance = bit_error_rate(decoded_messages[idx], gt_message)
        data[idx] = distance

    save_json(data, json_path)
    if not quiet:
        print(f"Decoded distances saved to {json_path}")


def main(picture_paths, message_paths, output_path, quiet=True):
    if not quiet:
        print(f"Evaluating stegastamp")
    assert len(picture_paths) == len(message_paths)
    decoded_messages = process(picture_paths, quiet)
    report(output_path, decoded_messages, picture_paths, message_paths, quiet)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Beige box decoding.")
    parser.add_argument(
        "-p",
        "--picture",
        nargs="+",
        dest="picture_paths",
        required=True,
        help="Path to the pictures",
    )
    parser.add_argument(
        "-m",
        "--message",
        nargs="+",
        dest="message_paths",
        required=True,
        help="Path to the ground-truth messages",
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
        picture_paths=args.picture_paths,
        message_paths=args.message_paths,
        output_path=args.output_path,
    )
