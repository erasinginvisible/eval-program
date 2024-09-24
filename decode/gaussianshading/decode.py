import os
import torch
from huggingface_hub import constants

current_dir = os.getcwd()
torch.hub.set_dir(os.path.join(current_dir, "cache", "pytorch"))
os.environ["TRANSFORMERS_CACHE"] = os.path.join(current_dir, "cache", "huggingface")
os.environ["HF_HOME"] = os.path.join(current_dir, "cache", "huggingface")
constants.HUGGINGFACE_HUB_CACHE = os.path.join(current_dir, "cache", "huggingface")

import warnings
import argparse
import json
from tqdm.auto import tqdm
import pickle
from PIL import Image
from modules.inverse_stable_diffusion import InversableStableDiffusionPipeline
from diffusers import DPMSolverMultistepScheduler
from modules.optim_utils import *
from modules.io_utils import *
from modules.image_utils import *
from modules.watermark import *

warnings.filterwarnings("ignore")


def save_json(data, filepath):
    with open(filepath, "w") as json_file:
        json.dump(data, json_file)


def load_model():
    scheduler = DPMSolverMultistepScheduler.from_pretrained(
        "stabilityai/stable-diffusion-2-1-base", subfolder="scheduler"
    )
    pipe = InversableStableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1-base",
        scheduler=scheduler,
        torch_dtype=torch.float16,
        revision="fp16",
    )
    pipe.safety_checker = None
    pipe = pipe.to("cuda")
    tester_prompt = ""
    text_embeddings = pipe.get_text_embedding(tester_prompt)
    return pipe, text_embeddings


def main(picture_paths, message_paths, output_path, quiet=True):
    if not quiet:
        print(f"Evaluating gaussianshading")
    assert len(picture_paths) == len(message_paths)

    pipe, text_embeddings = load_model()

    data = {}
    for picture_path, message_path in tqdm(
        zip(picture_paths, message_paths),
        total=len(picture_paths),
        desc="Decoding images",
    ):
        gt_message = pickle.load(open(message_path, "rb"))
        watermark = Gaussian_Shading_chacha(
            ch_factor=1,
            hw_factor=8,
            fpr=0.000001,
            user_number=1000000,
            watermark=gt_message,
        )
        image_w_distortion = (
            transform_img(Image.open(picture_path))
            .unsqueeze(0)
            .to(text_embeddings.dtype)
            .to("cuda")
        )
        image_latents_w = pipe.get_image_latents(image_w_distortion, sample=False)
        reversed_latents_w = pipe.forward_diffusion(
            latents=image_latents_w,
            text_embeddings=text_embeddings,
            guidance_scale=1,
            num_inference_steps=50,
        )
        _, acc_metric = watermark.eval_watermark(reversed_latents_w)
        distance = 1 - acc_metric
        data[int(picture_path.split("/")[-1].split(".")[0])] = distance

    json_path = os.path.join(output_path, "gaussianshading-decode.json")
    save_json(data, json_path)
    if not quiet:
        print(f"Decoded distances saved to {json_path}")


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
