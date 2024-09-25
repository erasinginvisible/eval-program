import os
import numpy as np

from utils import *


def decode(proc_dir, output_dir):
    logger = logging.getLogger("eval")

    image_paths_dict = {mode: [] for mode in THRESHOLD_FUNCTIONS.keys()}
    message_paths_dict = {mode: [] for mode in THRESHOLD_FUNCTIONS.keys()}

    for mode, shuffle_indices in SHUFFLE_DICT.items():
        if "+" not in mode:
            image_paths_dict[mode] += [
                str(proc_dir / f"{i}.png") for i in shuffle_indices
            ]
            message_paths_dict[mode] += sorted_listdir(
                f"data/encoded/{mode}/messages",
                ".txt" if mode not in ["gaussianshading", "prc"] else ".pkl",
            )
        else:
            for _mode in mode.split("+"):
                image_paths_dict[_mode] += [
                    str(proc_dir / f"{i}.png") for i in shuffle_indices
                ]
                message_paths_dict[_mode] += sorted_listdir(
                    f"data/encoded/{mode}/messages/{_mode}",
                    ".txt" if _mode not in ["gaussianshading", "prc"] else ".pkl",
                )

    for mode in image_paths_dict.keys():
        run_script(
            mode,
            image_paths_dict[mode],
            message_paths_dict[mode],
            output_dir,
        )

    logger.info("All decoding operations completed.")


def compute_performance(output_dir):
    logger = logging.getLogger("eval")

    # Process data
    results = []
    for mode, shuffle_indices in SHUFFLE_DICT.items():
        if "+" not in mode:
            distance_data = load_json(os.path.join(output_dir, f"{mode}-decode.json"))
            for idx in shuffle_indices:
                distance = distance_data[str(idx)]
                results.append(THRESHOLD_FUNCTIONS[mode][0](distance))
        else:
            mode1, mode2 = mode.split("+")
            distance_data1 = load_json(os.path.join(output_dir, f"{mode1}-decode.json"))
            distance_data2 = load_json(os.path.join(output_dir, f"{mode2}-decode.json"))
            for idx in shuffle_indices:
                distance1 = distance_data1[str(idx)]
                distance2 = distance_data2[str(idx)]
                results.append(
                    THRESHOLD_FUNCTIONS[mode1][1](distance1)
                    or THRESHOLD_FUNCTIONS[mode2][1](distance2)
                )

    logger.info("Performance calculation completed.")
    return np.mean(results)


def main(input_dir, output_dir, quiet=False):
    logger = setup_logger(quiet)

    try:
        verify_input_dir(input_dir)
        proc_dir = process_images("black", input_dir)
        decode(proc_dir, output_dir)
        metric("black", proc_dir, output_dir)
        performance = compute_performance(output_dir)
        quality = compute_quality(output_dir)
        save_results(output_dir, performance, quality)

        logger.info("All operations completed successfully.")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    args = parse_args()
    main(args.input_dir, args.output_dir)
