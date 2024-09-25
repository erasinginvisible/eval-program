import os
import torch

current_dir = os.getcwd()
torch.hub.set_dir(os.path.join(current_dir, "cache", "pytorch"))

import warnings
import argparse
import json
from tqdm import tqdm
from PIL import Image
import torch.nn as nn
import torchvision

from utils import *


warnings.filterwarnings("ignore")


def WMDecoder():
    decoder = torchvision.models.mobilenet_v3_large()
    proj_head = nn.Sequential(
        nn.Linear(960, 1280, bias=True), nn.Hardswish(), nn.Linear(1280, 1, bias=True)
    )
    decoder.classifier = proj_head
    return decoder


def load_model(device, model_path="./model/jigmark.pth"):
    decoder = WMDecoder()
    replace_batchnorm(decoder)
    checkpoint = torch.load(model_path, map_location=device)
    decoder.load_state_dict(checkpoint["decoder"])
    decoder = decoder.to(device)
    decoder.eval()
    return decoder


def main(args):
    imgs = [Image.open(image).resize((512, 512)) for image in args.img_paths]
    msgs = []
    for msg in args.msg_paths:
        with open(msg, "r") as f:
            msgs += [
                list(map(int, f.read().split(","))),
            ]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.cuda.set_device(0)  # Specify the GPU if needed

    decoder = load_model(device)

    transform = torchvision.transforms.ToTensor()
    transform_indices = [2, 0, 1, 3, 1, 0, 3, 4, 3, 4, 3, 1, 0, 2, 4, 0]
    results = []
    for n in tqdm(range(len(imgs))):
        # Load the shuffler
        shuffler = ImageShuffler(
            splits=4,
            shuffle_indices=msgs[n],
            transform_indices=transform_indices,
        )
        ex = transform(imgs[n]).unsqueeze(0).to(device)
        ex_value = decoder(shuffler.shuffle(ex))
        results.append(float(ex_value[0][0].item()))

    result_dict = {
        os.path.splitext(os.path.basename(img_path))[0]: result
        for img_path, result in zip(args.img_paths, results)
    }
    with open(os.path.join(args.output_dir, "jigmark-decode.json"), "w") as file:
        json.dump(result_dict, file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Args")
    parser.add_argument("-p", "--img_paths", nargs="+", type=str, required=True)
    parser.add_argument("-m", "--msg_paths", nargs="+", type=str, required=True)
    parser.add_argument("-o", "--output_dir", type=str, default="./data/result")
    args = parser.parse_args()
    main(args)
