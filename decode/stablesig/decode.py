import os
import torch
from huggingface_hub import constants

current_dir = os.getcwd()
torch.hub.set_dir(os.path.join(current_dir, "cache", "pytorch"))

import argparse
import json
from tqdm.auto import tqdm
from PIL import Image
from torchvision import transforms
import warnings

from utils import *


warnings.filterwarnings("ignore")

NORMALIZE_IMAGENET = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)


def load_model(device):
    params = Params(
        encoder_depth=4,
        encoder_channels=64,
        decoder_depth=8,
        decoder_channels=64,
        num_bits=48,
        attenuation="jnd",
        scale_channels=False,
        scaling_i=1,
        scaling_w=1.5,
    )
    decoder = HiddenDecoder(
        num_blocks=params.decoder_depth,
        num_bits=params.num_bits,
        channels=params.decoder_channels,
    )

    state_dict = torch.load("./model/stablesig.pth", map_location=device)[
        "encoder_decoder"
    ]
    encoder_decoder_state_dict = {
        k.replace("module.", ""): v for k, v in state_dict.items()
    }
    decoder_state_dict = {
        k.replace("decoder.", ""): v
        for k, v in encoder_decoder_state_dict.items()
        if "decoder" in k
    }
    decoder.load_state_dict(decoder_state_dict)
    decoder = decoder.to(device).eval()

    return decoder


def decode(model, image, device):
    default_transform = transforms.Compose([transforms.ToTensor(), NORMALIZE_IMAGENET])
    ft = model(default_transform(image).unsqueeze(0).to(device))
    return ft > 0


def bit_error_rate(pred, target, device, num_bits=48):
    msg_ori = torch.Tensor(str2msg(target)).unsqueeze(0)
    accs = ~torch.logical_xor(pred, msg_ori.to(device))
    accuracy = accs.sum().item() / num_bits
    return 1 - accuracy


def load_files(img_msg_dict, test=False):
    imgs = [Image.open(image_path).convert("RGB") for image_path in img_msg_dict.keys()]

    msgs = []
    for message_path in img_msg_dict.values():
        with open(message_path, "r") as file:
            binary_string = file.read()
            msgs.append(binary_string)

    if test:
        imgs = imgs + imgs
        msgs = msgs + msgs[::-1]

    return imgs, msgs


def main(args):
    imgs = [Image.open(image_path).convert("RGB") for image_path in args.img_paths]
    msgs = []
    for msg_path in args.msg_paths:
        with open(msg_path, "r") as file:
            binary_string = file.read()
            msgs.append(binary_string)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    decoder = load_model(device)

    outputs = [decode(decoder, img, device=device) for img in tqdm(imgs)]
    results = [
        bit_error_rate(output.flatten(), message, device=device)
        for output, message in zip(outputs, msgs)
    ]

    result_dict = {
        os.path.splitext(os.path.basename(img_path))[0]: result
        for img_path, result in zip(args.img_paths, results)
    }
    with open(os.path.join(args.output_dir, "stablesig-decode.json"), "w") as file:
        json.dump(result_dict, file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Args")
    parser.add_argument("-p", "--img_paths", nargs="+", type=str, required=True)
    parser.add_argument("-m", "--msg_paths", nargs="+", type=str, required=True)
    parser.add_argument("-o", "--output_dir", type=str, default="./data/result")
    args = parser.parse_args()
    main(args)
