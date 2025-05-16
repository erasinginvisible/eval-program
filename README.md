# Evaluation Program for Invisible Watermark Detection

This repository contains the evaluation program used for the beige-box and black-box tracks of the invisible watermark detection challenge. It is designed to be executed by a Codabench compute worker.

## Overview

The program takes a directory of 300 watermarked images (named `0.png` to `299.png`) as input and performs the following steps:

1.  **Verification:** Checks if the input directory contains the required 300 PNG files.
2.  **Image Processing:** Applies standard transformations (median blur, JPEG compression) to simulate common image handling, saving results to a temporary `proc` directory.
3.  **Decoding:** Attempts to decode watermarks embedded using various techniques.
4.  **Metric Calculation:** Computes image quality metrics (e.g., FID, PSNR, SSIM, LPIPS) between the processed images and original reference images.
5.  **Performance Calculation:** Determines the success rate of watermark detection across the different techniques.
6.  **Scoring:** Calculates final performance and quality scores based on the decoding results and metrics.
7.  **Output:** Saves the final scores to `scores.txt` in the designated output directory.

## Tracks

This program supports two evaluation tracks, executed via separate entry points:

*   **Beige Box (`beige.py`):** Evaluates specific known watermarking techniques (e.g., StegaStamp, Gaussian Shading).
*   **Black Box (`black.py`):** Evaluates detection performance across a mix of known and potentially unknown watermarking techniques, including combinations. *[NOTE FOR REVIEW: This track includes evaluation against the 'Trufo' watermark. Ensure this is appropriate for public release or remove/modify references in `utils.py` and `black.py` if needed.]*

## Repository Structure

```
eval-program/
├── beige.py             # Entry point for Beige Box track
├── black.py             # Entry point for Black Box track
├── utils.py             # Helper functions (I/O, processing, metrics, etc.)
├── requirements.txt     # Python dependencies
├── data/                # Likely contains reference data, messages, etc. (Not provided in detail)
├── decode/              # Scripts/logic for decoding different watermarks (Not provided in detail)
├── metric/              # Scripts/logic for calculating image quality metrics (Not provided in detail)
├── model/               # Models used for evaluation (e.g., metric calculation) (Not provided in detail)
├── front/               # Unknown purpose (Not provided in detail)
├── test/                # Tests for the evaluation program (Not provided in detail)
├── cache/               # Directory for caching (e.g., model downloads)
└── .gitignore           # Git ignore configuration
```

## Prerequisites

*   **Python:** Python 3.x
*   **Dependencies:** Install required packages using pip:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: `requirements.txt` includes `onnxruntime-gpu`, implying GPU usage is expected for efficient execution, particularly for metric calculations. Ensure appropriate CUDA/GPU drivers are installed if running locally.*
*   **Data:** The program expects certain data files (e.g., reference images, messages) potentially within the `data/` directory structure, which are not fully detailed here but are referenced by the code (e.g., `data/encoded/...`).

## Usage (Local Execution)

While designed for Codabench, you can run the evaluation locally using the entry point scripts:

```bash
# For Beige Box Track
python beige.py --input_dir /path/to/your/submission/images --output_dir /path/to/output

# For Black Box Track
python black.py --input_dir /path/to/your/submission/images --output_dir /path/to/output
```

*   Replace `/path/to/your/submission/images` with the directory containing the 300 input PNG files (`0.png` - `299.png`). The script can handle cases where these files are inside a single sub-directory within the specified input directory.
*   Replace `/path/to/output` with the directory where results (including `scores.txt`) should be saved.

## Integration with Codabench

This program is intended to be packaged within a Docker container (like the one described in the `worker-container` repository) and run by a Codabench compute worker. The worker executes the appropriate script (`beige.py` or `black.py`) providing the submission and output directories as arguments.

## License
```
Copyright [2025] [Mucong Ding]

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
