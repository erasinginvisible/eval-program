import os

os.environ["TQDM_MININTERVAL"] = "5"
os.environ["TQDM_POSITION"] = "-1"


from utils import *


def decode(proc_dir, output_dir):
    logger = logging.getLogger("eval")

    run_script(
        "stegastamp",
        [str(proc_dir / f"{i}.png") for i in range(150)],
        sorted_listdir("data/encoded/stegastamp/messages", ".txt"),
        output_dir,
    )
    run_script(
        "gaussianshading",
        [str(proc_dir / f"{i}.png") for i in range(150, 300)],
        sorted_listdir("data/encoded/gaussianshading/messages", ".pkl"),
        output_dir,
    )

    logger.info("All decoding operations completed.")


def compute_performance(output_dir):
    logger = logging.getLogger("eval")

    # Load JSON files
    stegastamp_data = load_json(os.path.join(output_dir, "e-decode.json"))
    gaussianshading_data = load_json(os.path.join(output_dir, "a-decode.json"))

    # Process data
    results = []
    for idx in range(150):
        distance = stegastamp_data[str(idx)]
        results.append(THRESHOLD_FUNCTIONS["stegastamp"][0](distance))
    for idx in range(150, 300):
        distance = gaussianshading_data[str(idx)]
        results.append(THRESHOLD_FUNCTIONS["gaussianshading"][0](distance))

    logger.info("Performance calculation completed.")
    return np.mean(results)


def main(input_dir, output_dir, quiet=False):
    logger = setup_logger(quiet)
    logger.info("Evaluation program started.")

    try:
        verify_input_dir(input_dir, output_dir)
        proc_dir = process_images("beige", input_dir)
        decode(proc_dir, output_dir)
        metric("beige", proc_dir, output_dir)
        performance = compute_performance(output_dir)
        quality = compute_quality(output_dir)
        save_results(output_dir, performance, quality)
        remove_results(input_dir, output_dir)

        logger.info("All operations completed successfully.")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    args = parse_args()
    main(args.input_dir, args.output_dir)
