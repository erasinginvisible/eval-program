import os
import subprocess
import warnings
import argparse
import json
import glob
import numpy as np

warnings.filterwarnings("ignore")

SHUFFLE_DICT = {
    "jigmark": [
        73,
        213,
        173,
        106,
        59,
        253,
        168,
        26,
        284,
        153,
        134,
        145,
        63,
        293,
        285,
        224,
        252,
        111,
        20,
        46,
        156,
        228,
        273,
        27,
        144,
        259,
        37,
        97,
        191,
        135,
        118,
        160,
        264,
        214,
        238,
        76,
        212,
        225,
        255,
        237,
        282,
        44,
        272,
        189,
        152,
        158,
        101,
        54,
        181,
        18,
    ],
    "prc": [
        281,
        250,
        126,
        171,
        71,
        227,
        245,
        205,
        288,
        215,
        154,
        159,
        33,
        83,
        249,
        60,
        167,
        280,
        110,
        21,
        29,
        146,
        16,
        56,
        75,
        109,
        175,
        201,
        161,
        4,
        96,
        166,
        61,
        67,
        137,
        198,
        262,
        279,
        40,
        268,
        13,
        107,
        220,
        3,
        157,
        125,
        24,
        30,
        77,
        291,
    ],
    "stablesig": [
        210,
        19,
        254,
        241,
        266,
        80,
        51,
        2,
        235,
        104,
        179,
        86,
        10,
        199,
        58,
        41,
        14,
        155,
        50,
        292,
        233,
        123,
        200,
        62,
        187,
        226,
        130,
        209,
        260,
        43,
        114,
        138,
        294,
        218,
        149,
        112,
        247,
        98,
        217,
        93,
        216,
        162,
        36,
        178,
        113,
        0,
        94,
        275,
        95,
        296,
    ],
    "trufo": [
        79,
        17,
        263,
        232,
        38,
        133,
        53,
        258,
        128,
        28,
        183,
        163,
        151,
        244,
        202,
        31,
        32,
        127,
        185,
        278,
        271,
        147,
        276,
        177,
        99,
        197,
        243,
        115,
        265,
        72,
        25,
        165,
        287,
        174,
        289,
        39,
        193,
        88,
        70,
        87,
        242,
        277,
        211,
        9,
        195,
        251,
        192,
        117,
        47,
        172,
    ],
    "stablesig+stegastamp": [
        261,
        240,
        69,
        49,
        48,
        85,
        297,
        141,
        207,
        23,
        246,
        148,
        143,
        78,
        180,
        100,
        204,
        131,
        267,
        298,
        196,
        6,
        68,
        203,
        84,
        170,
        121,
        140,
        256,
        274,
        142,
        257,
        91,
        82,
        283,
        11,
        119,
        102,
        35,
        57,
        169,
        231,
        65,
        1,
        120,
        269,
        186,
        42,
        105,
        132,
    ],
    "gaussianshading+jigmark": [
        208,
        188,
        12,
        221,
        239,
        136,
        230,
        206,
        52,
        108,
        290,
        15,
        184,
        248,
        22,
        74,
        270,
        90,
        229,
        164,
        190,
        103,
        124,
        129,
        222,
        116,
        194,
        286,
        234,
        295,
        92,
        66,
        8,
        122,
        219,
        150,
        176,
        182,
        299,
        5,
        223,
        89,
        81,
        34,
        55,
        139,
        236,
        64,
        7,
        45,
    ],
}
GT_PATH = "./data/encoded"


def sorted_listdir(msg_dir, ext):
    ori_list = glob.glob(f"{msg_dir}/*{ext}")
    if len(ori_list) == 0:
        return []
    else:
        indices = [
            int(os.path.splitext(os.path.basename(path))[0]) for path in ori_list
        ]
        return [os.path.join(msg_dir, f"{idx}{ext}") for idx in sorted(indices)]


def decode_command(decode_type, img_queue, msg_queue, output_path):
    assert len(img_queue) == len(msg_queue)
    command = (
        ["python", f"decode/{decode_type}/decode.py", "-p"]
        + img_queue
        + ["-m"]
        + msg_queue
        + ["-o", output_path]
    )
    subprocess.run(command)


def check_result(output_path):
    bool_result_collection = {
        wm_type: None
        for wm_type in [
            "jigmark",
            "trufo",
            "stablesig",
            "stegastamp",
            "gaussianshading",
            "prc",
        ]
    }
    for wm_type, threshold_function in {
        "jigmark": [lambda x: float(x > 3.27), lambda x: float(x > 3.3)],
        "trufo": [lambda x: float(x < 0.3167)],
        "stablesig": [lambda x: float(x < 0.29167), lambda x: float(x < 0.28)],
        "stegastamp": [lambda x: float(x < 0.34)],
        "gaussianshading": [lambda x: float(x < 0.402)],
        "prc": [lambda x: x],
    }.items():
        if f"{wm_type}-decode.json" in os.listdir(output_path):
            with open(os.path.join(output_path, f"{wm_type}-decode.json")) as json_data:
                indices_distances = json.load(json_data)
                distances = [float(distance) for distance in indices_distances.values()]
                bool_result_collection[wm_type] = [
                    threshold_function[0](distance) for distance in distances[0:50]
                ]
                if len(distances) > 50:
                    bool_result_collection[wm_type] += [
                        threshold_function[1](distance)
                        for distance in distances[50:100]
                    ]
    bool_result_collection["gaussianshading"] = [
        a or b
        for a, b in zip(
            bool_result_collection["gaussianshading"],
            bool_result_collection["jigmark"][50:100],
        )
    ]
    bool_result_collection["stegastamp"] = [
        a or b
        for a, b in zip(
            bool_result_collection["stegastamp"],
            bool_result_collection["stablesig"][50:100],
        )
    ]
    final_result = []
    for wm_type in [
        "jigmark",
        "trufo",
        "stablesig",
        "stegastamp",
        "gaussianshading",
        "prc",
    ]:
        final_result += bool_result_collection[wm_type][0:50]
    print(np.mean(final_result))


def main(args):
    img_paths = sorted_listdir(args.submit_path, ".png")
    img_path_queues = {
        wm_type: []
        for wm_type in [
            "jigmark",
            "trufo",
            "stablesig",
            "stegastamp",
            "gaussianshading",
            "prc",
        ]
    }
    msg_path_queues = {
        wm_type: []
        for wm_type in [
            "jigmark",
            "trufo",
            "stablesig",
            "stegastamp",
            "gaussianshading",
            "prc",
        ]
    }
    for wm_type, shuffle_indices in SHUFFLE_DICT.items():
        img_queue = [img_paths[idx] for idx in shuffle_indices]
        if "+" in wm_type:
            for wm_subtype in wm_type.split("+"):
                img_path_queues[wm_subtype] += img_queue
                msg_dir = os.path.join(GT_PATH, wm_type, "messages", wm_subtype)
                msg_path_queues[wm_subtype] += sorted_listdir(
                    msg_dir,
                    ".pkl" if wm_subtype in ["gaussianshading", "prc"] else ".txt",
                )
        else:
            img_path_queues[wm_type] += img_queue
            msg_dir = os.path.join(GT_PATH, wm_type, "messages")
            msg_path_queues[wm_type] += sorted_listdir(
                msg_dir, ".pkl" if wm_type in ["gaussianshading", "prc"] else ".txt"
            )

    for wm_type in img_path_queues.keys():
        decode_command(
            wm_type,
            img_path_queues[wm_type],
            msg_path_queues[wm_type],
            args.output_path,
        )

    check_result(args.output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Args")
    parser.add_argument("-i", "--submit_path", type=str, required=True)
    parser.add_argument("-o", "--output_path", type=str, required=True)
    args = parser.parse_args()
    main(args)
