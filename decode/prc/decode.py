import os
import torch
from huggingface_hub import constants

current_dir = os.getcwd()
torch.hub.set_dir(os.path.join(current_dir, "cache", "pytorch"))
os.environ["TRANSFORMERS_CACHE"] = os.path.join(current_dir, "cache", "huggingface")
os.environ["HF_HOME"] = os.path.join(current_dir, "cache", "huggingface")
constants.HUGGINGFACE_HUB_CACHE = os.path.join(current_dir, "cache", "huggingface")

import argparse
import pickle
import warnings
import json
from tqdm.auto import tqdm
from PIL import Image
import numpy as np
from diffusers import DPMSolverMultistepScheduler

from utils import Detect, Decode, prc_gaussians, transform_img
from modules.inverse_vae import InversableVAE
from modules.inverse_stable_diffusion import InversableStableDiffusionPipeline

warnings.filterwarnings("ignore")


def stable_diffusion_pipe(
    solver_order=1,
    model_id="stabilityai/stable-diffusion-2-1-base",
):
    # load stable diffusion pipeline
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scheduler = DPMSolverMultistepScheduler(
        beta_end=0.012,
        beta_schedule="scaled_linear",
        beta_start=0.00085,
        num_train_timesteps=1000,
        prediction_type="epsilon",
        steps_offset=1,
        trained_betas=None,
        solver_order=solver_order,
    )
    pipe = InversableStableDiffusionPipeline.from_pretrained(
        model_id,
        scheduler=scheduler,
        torch_dtype=torch.float32,
    )
    pipe = pipe.to(device)

    return pipe


def exact_inversion_vae(
    images,
    vae,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # image to latent
    images_tensor = (
        torch.stack([transform_img(img) for img in images]).to(torch.float32).to(device)
    )
    image_latents = vae.decoder_inv(images_tensor)

    return image_latents


def load_vae(checkpoint_path="./model/prc.pt", device="cuda"):
    vae = InversableVAE(
        down_block_types=(
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
        ),
        up_block_types=(
            "UpDecoderBlock2D",
            "UpDecoderBlock2D",
            "UpDecoderBlock2D",
            "UpDecoderBlock2D",
        ),
        block_out_channels=(128, 256, 512, 512),
        layers_per_block=2,
    ).to(device)

    vae_dict = torch.load(checkpoint_path)
    vae.load_state_dict(vae_dict)
    return vae


def load_pipe(model_id="stabilityai/stable-diffusion-2-1-base", solver_order=1):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scheduler = DPMSolverMultistepScheduler(
        beta_end=0.012,
        beta_schedule="scaled_linear",
        beta_start=0.00085,
        num_train_timesteps=1000,
        prediction_type="epsilon",
        steps_offset=1,
        trained_betas=None,
        solver_order=solver_order,
    )
    pipe = InversableStableDiffusionPipeline.from_pretrained(
        model_id,
        scheduler=scheduler,
        torch_dtype=torch.float32,
    )
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def decode_batch_vae(vae, images, batch_size=1):
    image_latents = []
    for idx in tqdm(range(0, len(images), batch_size)):
        image_latents += exact_inversion_vae(
            images[idx * batch_size : min((idx + 1) * batch_size, len(images))],
            vae,
        )
    return image_latents


def decode_batch_rest(
    pipe,
    image_latents,
    decoding_keys,
    guidance_scale=3.0,
    test_num_inference_steps=50,
    batch_size=5,
):
    cur_inv_order = 0
    var = 1.5

    # prompt to text embeddings
    text_embeddings_tuple = pipe.encode_prompt(
        [
            "",
        ]
        * batch_size,
        "cuda",
        1,
        guidance_scale > 1.0,
        None,
    )
    text_embeddings = torch.cat([text_embeddings_tuple[1], text_embeddings_tuple[0]])

    reversed_latents = []
    for idx in tqdm(np.arange(0, len(image_latents), batch_size)):
        reversed_latent_batch = pipe.forward_diffusion(
            latents=torch.stack(
                image_latents[idx : min(idx + batch_size, len(image_latents))]
            ),
            text_embeddings=text_embeddings,
            guidance_scale=guidance_scale,
            num_inference_steps=test_num_inference_steps,
            inverse_opt=(cur_inv_order != 0),
            inv_order=cur_inv_order,
        )
        reversed_latents += reversed_latent_batch

    results = []
    for reversed_latent, key in zip(reversed_latents, decoding_keys):
        reversed_prc = (
            prc_gaussians(
                reversed_latent.to(torch.float64).flatten().cpu(), variances=float(var)
            )
            .flatten()
            .cpu()
        )
        detection_result = Detect(key, reversed_prc)
        decoding_result = Decode(key, reversed_prc) is not None
        combined_result = detection_result or decoding_result
        results.append(float(combined_result))
    return results


def main(args):
    imgs = [Image.open(image) for image in args.img_paths]
    msgs = []
    for msg in args.msg_paths:
        with open(msg, "rb") as f:
            _, decoding_key = pickle.load(f)
            msgs.append(decoding_key)

    vae = load_vae()
    image_latents = decode_batch_vae(vae, imgs)

    pipe = load_pipe()
    results = decode_batch_rest(pipe, image_latents, msgs, test_num_inference_steps=50)

    result_dict = {
        os.path.splitext(os.path.basename(img_path))[0]: result
        for img_path, result in zip(args.img_paths, results)
    }
    with open(os.path.join(args.output_dir, "prc-decode.json"), "w") as file:
        json.dump(result_dict, file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Args")
    parser.add_argument("-p", "--img_paths", nargs="+", type=str, required=True)
    parser.add_argument("-m", "--msg_paths", nargs="+", type=str, required=True)
    parser.add_argument("-o", "--output_dir", type=str, default="./data/result")
    args = parser.parse_args()
    main(args)
