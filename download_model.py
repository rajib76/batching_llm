"""
Script to download the GPT-2 model and tokenizer from Hugging Face and save them locally.
This allows the other scripts to run without needing an internet connection every time
or if the local path is preferred.
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import os

def download_and_save_model(model_name="gpt2", save_dir="./models/gpt2"):
    """
    Downloads the specified model and tokenizer and saves them to the given directory.

    Args:
        model_name (str): The name of the model to download (default: "gpt2").
        save_dir (str): The directory to save the model and tokenizer to.
    """
    print(f"Downloading {model_name}...")
    
    # Create the directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Download and save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(save_dir)
    print(f"Tokenizer saved to {save_dir}")

    # Download and save model
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.save_pretrained(save_dir)
    print(f"Model saved to {save_dir}")

if __name__ == "__main__":
    download_and_save_model()
