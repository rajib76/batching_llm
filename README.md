# Batching and KV Caching with LLMs

This repository demonstrates the concepts of batching and Key-Value (KV) caching when working with Large Language Models (LLMs), specifically GPT-2, using the Hugging Face `transformers` library.

## Project Structure

- `inference_wrapper.py`: Defines a `Batching` class that encapsulates the model and tokenizer, providing a method for generating tokens.
- `download_model.py`: A helper script to download and save the GPT-2 model locally. This avoids repeated downloads.
- `naive_generation.py`: **Naive Generation**. Demonstrates token generation *without* KV caching. This is inefficient because the model re-processes the entire history at each step.
- `kv_cache_generation.py`: **Efficient Generation**. Demonstrates token generation *with* KV caching. This is much faster because it reuses the computation from previous steps.

## Setup

1.  **Install Requirements:**
    Make sure you have Python installed. Install the necessary libraries using pip:

    ```bash
    pip install -r requirements.txt
    ```

    *Note: You may need to create a virtual environment first.*

2.  **Download Model:**
    Run the download script to fetch the GPT-2 model and save it to `./models/gpt2`:

    ```bash
    python download_model.py
    ```

## Usage

### 1. Naive Generation (Slow)

Run `naive_generation.py` to see how text generation works without optimization. Observe the time it takes.

```bash
python naive_generation.py
```

Process:
- Takes a prompt.
- Generates new tokens one by one.
- In each step, concatenates the new token to the *full* input sequence.
- Re-runs the model on the growing sequence.
- **Complexity:** O(N^2)

### 2. Efficient Generation with KV Cache (Fast)

Run `kv_cache_generation.py` to see the performance improvement using KV caching.

```bash
python kv_cache_generation.py
```

Process:
- Takes a prompt.
- In the first step, processes the full prompt and gets `past_key_values`.
- For subsequent steps, passes only the *new* token and the cached `past_key_values`.
- The model only computes attention for the new token.
- **Complexity:** O(N)

## Key Concepts

### KV Caching
In Transformer models, the Attention mechanism computes "Keys" and "Values" for each token. When generating text sequentially, the Keys and Values for past tokens don't change. By caching them (`past_key_values` in Hugging Face), we avoid re-calculating them, leading to significant speedups, especially for long sequences.

### Batching (referenced in `inference_wrapper.py`)
While `inference_wrapper.py` sets up the structure, the examples primarily focus on the sequential generation part. Batching usually involves processing multiple distinct sequences in parallel to maximize GPU utilization.
