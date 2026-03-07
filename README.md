# SinkTrack: Attention Sink based Context Anchoring for Large Language Models

The official implementation of "[SinkTrack: Attention Sink based Context Anchoring for Large Language Models](https://openreview.net/forum?id=Gg1aPETCL6)" (ICLR 2026).

## Abstract
Large language models (LLMs) suffer from hallucination and context forgetting. These problems are caused by attention drift, where LLMs’ focus shifts towards newly generated tokens and away from the initial input context. To address this, we make use of a related, intrinsic characteristic of LLMs: attention sink – the tendency to consistently allocate high attention to the very first token (i.e., ⟨BOS⟩) of a sequence. Concretely, we propose an advanced context anchoring method, SINKTRACK, which treats ⟨BOS⟩ as an information anchor and injects key contextual features (such as those derived from the input image or instruction) into its representation. As such, LLM remains anchored to the initial input context throughout the entire generation process. SINKTRACK is training-free, plug-and-play, and introduces negligible inference overhead. Experiments demonstrate that SINKTRACK mitigates hallucination and context forgetting across both textual (e.g., +18.9% on QuAC with Llama3.1-8B-Instruct) and multi-modal (e.g., +23.0% on M3CoT with Qwen2.5-VL-7B-Instruct) tasks. Its consistent gains across different architectures and scales underscore the robustness and generalizability. We also analyze its underlying working mechanism from the perspective of information delivery.

---

## Installation

Follow these steps to set up the environment and install the required dependencies.

1.  **Create and activate the Conda environment**
    ```bash
    # Create the conda environment with Python 3.10
    conda create -n sinkTrack python=3.10

    # Activate the newly created environment
    conda activate sinkTrack
    ```

2.  **Install dependencies**
    ```bash
    # Install all required packages from requirements.txt
    pip install -r requirements.txt
    ```

---

## 1. Reproducing Results: Qwen2.5-VL on M3CoT

**Note:** All operations in this section should be performed within the `all_inference_codes` directory unless otherwise specified.

This section guides you through the complete pipeline for running Qwen2.5-VL on the M3CoT dataset, including data preparation, inference, and evaluation.

### Step 1: Download Dataset
Download the M3CoT dataset from HuggingFace:
*   **URL:** https://huggingface.co/datasets/LightChen2333/M3CoT
*   **Destination:** Place the downloaded files into the `all_inference_codes/datasets` directory.

### Step 2: Process Dataset
We provide a script to format the data.
*   **Requirement:** The M3CoT dataset must contain the following keys: `id` (question id), `image` (image path), `question`, `choices`, and `answer`.
*   **Action:** Navigate to the datasets directory and run the processing script:
    ```bash
    cd datasets
    python process_m3cot.py
    ```
*   **Output:** This will generate:
    *   An image folder: `datasets/M3CoT/data/images` (images named by ID).
    *   A JSON file: `datasets/M3CoT/data/test.json` containing the required keys.

### Step 3: Download Model
Download the **Qwen2.5-VL-7B-Instruct** model.
*   **URL:** https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
*   **Destination:** Place the model files into the `all_inference_codes/models` directory.

### Step 4: Run Inference
**Important:** Ensure you are back in the `all_inference_codes` directory before proceeding.

Navigate to the Qwen2.5-VL directory and run the inference scripts for different methods (Direct, CoT, and SinkTrack).

```bash
cd qwen25vl

# Run Direct Inference
python direct.py

# Run Chain-of-Thought (CoT) Inference
python cot.py

# Run SinkTrack Inference
python run.py
```

The results will be saved in the following directories:
*   **Direct:** `qwen25vl/direct/qwen7b/m3cot`
*   **CoT:** `cot/qwen7b/m3cot`
*   **SinkTrack:** `sinktrack/qwen7b/m3cot`

### Step 5: Evaluate Results
To evaluate the performance of the generated results, run the evaluation script:

```bash
python eval.py
```
This will output the evaluation metrics for Direct, CoT, and SinkTrack methods.

---

## 2. Reproducing Results: Llama 3.1 on QuAC

**Note:** All operations in this section should be performed from the `all_inference_codes/llama31` directory (e.g. run `cd all_inference_codes/llama31` from the repo root). Paths in `config.yaml` are relative to the repo root and work both locally and on Google Colab.

This section guides you through the complete pipeline for running Llama 3.1-8B-Instruct on the QuAC dataset, including data preparation, inference, and evaluation.

### Step 1: Prepare Validation Data

*   **Requirement:** The validation JSON must follow the QuAC format (e.g. `val_v0.2.json` with a `data` key containing articles and paragraphs).
*   **Placement:** By default the scripts expect the file at `all_inference_results/llama3_1/quac/val_v0.2.json` (see `data.val_data_path` in `config.yaml`). Place your file there or update the path in `config.yaml`.

### Step 2: Download Model

Download the **Llama-3.1-8B-Instruct** model.
*   **URL:** https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct
*   **Destination:** Place the model files into the `all_inference_codes/models` directory (or the path set in `config.yaml` → `model.model_path`).

### Step 3: Run Inference

**Important:** Run the commands below from the `all_inference_codes/llama31` directory.

Navigate to the `llama31` directory and run the inference scripts for different methods (Direct, CoT, and SinkTrack).

```bash
cd all_inference_codes/llama31   # from repo root; omit if already there

# Run Direct Inference
python direct.py

# Run Chain-of-Thought (CoT) Inference
python cot.py

# Run SinkTrack Inference
python run.py
```

The results are saved in a single directory:
*   **Output directory:** `all_inference_results/llama3_1/quac/`
*   **Files:** `direct_323.jsonl`, `direct_500.jsonl`, `direct_900.jsonl`, and similarly `cot_*.jsonl` and `sinktrack_*.jsonl` for each method.

### Step 4: Evaluate Results

To evaluate the performance of the generated results, run the evaluation script:

```bash
python new_scorer.py
```

This will output the evaluation metrics (F1, HEQ, DHEQ, etc.) for the selected baseline across seeds. Use `--baseline direct`, `--baseline cot`, or the default SinkTrack; see `python new_scorer.py --help` for more options.

---

## 3. Customizing Models and Datasets

This section explains how to apply this framework to different datasets or models.

### Changing the Dataset
To use a custom dataset:
1.  **Format Requirements:** Ensure your dataset is processed to include the following keys:
    *   `id`: Unique question identifier.
    *   `image`: Path to the image file.
    *   `question`: The text query.
    *   `answer`: Ground truth answer.
    *   `choices`: (Optional) Required if the task is multiple-choice.
2.  **Placement:** Place the dataset in the `datasets` folder.
3.  **Configuration:** Open `direct.py`, `cot.py`, and `run.py`. Modify the dataset loading paths to point to your new dataset directory.

### Changing the Model
To use a different model:
1.  **Download:** Download the desired model and place it in the `models` directory.
2.  **Configuration:** Open `direct.py`, `cot.py`, and `run.py`. Update the model loading paths to point to the new model in the `models` directory.

---

## 4. Evaluating Pre-Uploaded Inference Results

If you wish to assess the performance of the inference results we have already uploaded, follow the instructions below based on the modality.

### A. Multimodal Dataset Evaluation
To evaluate SinkTrack results for a specific model on a specific dataset (across 3 random seeds: 323, 500, 900):

1.  Navigate to the specific results folder.
2.  Run `python t.py`.

**Example:** Evaluating **Gemma3-4B** using SinkTrack on the **MMStar** dataset.

```bash
cd all_inference_results/gemma3/sinktrack/4b/mmstar
python t.py
```
*Output:* This will display the results for each seed file, as well as the mean and variance across the three runs.

### B. Textual Dataset Evaluation
To evaluate all methods (Direct, CoT, SinkTrack) for a specific model on a textual dataset (across 3 random seeds: 323, 500, 900):

1.  Navigate to the specific dataset folder within the model directory.
2.  Run `python eval.py`.

**Example:** Evaluating **Llama3.1-8B-Instruct** on the **QuAC** dataset (results for Direct, CoT, and SinkTrack).

```bash
cd all_inference_results/llama3_1/quac
python eval.py
```
*Output:* This will display the evaluation metrics for all methods located in that directory.
