# 🚀 SinkTrack: Attention Sink based Context Anchoring for Large Language Models

---
## 📝 Abstract
Large language models (LLMs) suffer from hallucination and context forgetting. These problems are caused by attention drift, where LLMs’ focus shifts towards newly generated tokens and away from the initial input context. To address this, we make use of a related, intrinsic characteristic of LLMs: attention sink – the tendency to consistently allocate high attention to the very first token (i.e., ⟨BOS⟩) of a sequence. Concretely, we propose an advanced context anchoring method, SINKTRACK, which treats ⟨BOS⟩ as an information anchor and injects key contextual features (such as those derived from the input image or instruction) into its representation. As such, LLM remains anchored to the initial input context throughout the entire generation process. SINKTRACK is training-free, plug-and-play, and introduces negligible inference overhead. Experiments demonstrate that SINKTRACK mitigates hallucination and context forgetting across both textual (e.g., +18.9% on QuAC with Llama3.1-8B-Instruct) and multi-modal (e.g., +23.0% on M3CoT with Qwen2.5-VL-7B-Instruct) tasks. Its consistent gains across different architectures and scales underscore the robustness and generalizability. We also analyze its underlying working mechanism from the perspective of information delivery. 


---

## 📑 Table of Contents
This repository contains the code and instructions to run demos and benchmark models like **Qwen 2.5**.
- [⚙️ Installation](#️-installation)
- [🧪 Demo Usage](#-demo-usage)
  - [🖼️ Visual Question Answering (VQA) Demo](#️-visual-question-answering-vqa-demo)
  - [✍️ Text Generation Demo](#-text-generation-demo)
- [📊 Benchmarking](#-benchmarking)

---

## ⚙️ Installation

Follow these steps to set up the environment and install the required dependencies.

1.  **Clone the repository** 📂
    ```bash
    git clone <REPOSITORY_URL>
    cd <REPOSITORY_FOLDER>
    ```

2.  **Create and activate the Conda environment** 🐍
    ```bash
    # Create the conda environment with Python 3.10
    conda create -n sinkTrack python=3.10

    # Activate the newly created environment
    conda activate sinkTrack
    ```

3.  **Install dependencies** 📦
    ```bash
    # Install all required packages from requirements.txt
    pip install -r requirements.txt
    ```

---

## 🧪 Demo Usage

We provide two simple demo scripts to showcase the capabilities of the models.

### 🖼️ Visual Question Answering (VQA) Demo
This demo utilizes the **Qwen-2.5-VL** model to answer questions based on an image.

**Run the demo:**
```bash

python demo/vqa_qwen_vl_demo.py
````

### ✍️ Text Generation Demo

This demo showcases the text generation capabilities of the **Qwen-2.5** LLM.

**Run the demo:**

```bash
python demo/text_qwen_demo.py
```

---

## 📊 Benchmarking

This section provides a step-by-step guide to benchmark the model performance on the full **MM-Star** dataset.

### 🛠️ Step 1: Download the Dataset

First, download the MMStar dataset from its official source.

* **Source:** [MMStar](https://huggingface.co/datasets/Lin-Chen/MMStar)
* **Action:** Download the dataset and place it in `benchmarking_scripts/`.

---

### 🔧 Step 2: Prepare Data

Run the preprocessing script to extract images and convert the dataset into the required format.
Located in the `benchmarking_scripts/` directory:

```bash
python benchmarking_scripts/par_json.py
```

👉 This command will generate:

* an `images/` folder
* a `test.json` file inside `benchmarking_scripts/`

---

### 🤖 Step 3: Run Inference

Once the data is ready, run the benchmark script to generate predictions:

```bash
python benchmarking_scripts/run_benchmark.py
```

📂 Output: predictions saved in a results file (e.g., `results.json`).

---

### 📏 Step 4: Evaluate Results

Finally, compare predictions with ground truth to get evaluation metrics:

```bash
python benchmarking_scripts/evaluate.py
```

✅ The script will print the final evaluation scores directly to the console.

---
