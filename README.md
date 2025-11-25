# SinkTrack: Attention Sink based Context Anchoring for Large Language Models

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

This section guides you through the complete pipeline for running Qwen2.5-VL on the M3CoT dataset, including data preparation, inference, and evaluation.

### Step 1: Download Dataset
Download the M3CoT dataset from HuggingFace:
*   **URL:** https://huggingface.co/datasets/LightChen2333/M3CoT
*   **Destination:** Place the downloaded files into the `all_inference_codes/datasets` directory.

### Step 2: Process Dataset
We provide a script to format the data.
*   **Requirement:** The M3CoT dataset must contain the following keys: `id` (question id), `image` (image path), `question`, `choices`, and `answer`.
*   **Action:** Run the processing script:
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

## 2. Customizing Models and Datasets

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

## 3. Evaluating Pre-Uploaded Inference Results

If you wish to assess the performance of the inference results we have already uploaded, follow the instructions below based on the modality.

### A. Multimodal Dataset Evaluation
To evaluate SinkTrack results for a specific model on a specific dataset (across 3 random seeds: 323, 500, 900):

1.  Navigate to the specific results folder.
2.  Run `python t.py`.

**Example:** Evaluating **Gemma3-12B** using SinkTrack on the **MMStar** dataset.

```bash
cd all_inference_results/gemma3/sinktrack/12b/mmstar
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
