"""
Example script demonstrating naive token generation with GPT-2.
This script generates text token by token without using KV caching,
which means it re-computes the entire context for every new token.
It measures and prints the time taken for generation to demonstrate the inefficiency.
"""

import matplotlib.pyplot as plt
import numpy as np
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the locally saved model and tokenizer
model_name = "./models/gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Initial prompt
prompt = "The quick brown fox jumped over the"
inputs = tokenizer(prompt, return_tensors="pt")
# print(inputs)

# -----------------------------------------------------------------------------
# Demonstration of a single forward pass
# -----------------------------------------------------------------------------
with torch.no_grad():
    outputs = model(**inputs)

logits = outputs.logits
print(f"Logits shape: {logits.shape}") # (1, sequence_length, vocab_size)

# Extract logits for the last token
last_logits = logits[0, -1, :]
next_token_id = last_logits.argmax() # Greedy decoding
# print(tokenizer.decode(next_token_id))

# Show top 10 candidates
top_k = torch.topk(last_logits, k=10)
tokens = [tokenizer.decode(tk) for tk in top_k.indices]
print(f"Top 10 candidates: {tokens}")


# Prepare inputs for the next step by appending the new token
next_inputs = {
    "input_ids": torch.cat(
        [inputs["input_ids"], next_token_id.reshape((1, 1))],
        dim=1,
    ),
    "attention_mask": torch.cat(
        [inputs["attention_mask"], torch.tensor([[1]])],
        dim=1
    ),
}

# -----------------------------------------------------------------------------
# Function for generating a single token (naive approach)
# -----------------------------------------------------------------------------
def generate_token(inputs):
    """
    Generates the next token ID given the inputs.
    
    Args:
        inputs (dict): Dictionary containing 'input_ids' and 'attention_mask'.
                       Note: Pass the FULL sequence history here.
        
    Returns:
        torch.Tensor: The ID of the next token.
    """
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    # We only care about the prediction for the last token position
    last_logits = logits[0, -1, :]
    next_token_id = last_logits.argmax()
    return next_token_id


# -----------------------------------------------------------------------------
# Generation Loop
# -----------------------------------------------------------------------------
generated_tokens = []
next_inputs = inputs
duration_s = []

print("\nStarting generation loop...")
for _ in range(10):
    t0 = time.time()
    
    # Generate one token
    next_token_id = generate_token(next_inputs)
    
    duration_s.append(time.time() - t0)

    # Append the predicted token to the inputs for the next iteration
    # This naive approach grows the input sequence length by 1 each time
    # and re-processes the ENTIRE sequence, which is inefficient (O(N^2) complexity).
    next_inputs = {
        "input_ids": torch.cat(
            [next_inputs["input_ids"], next_token_id.reshape((1, 1))],
            dim=1),
        "attention_mask": torch.cat(
            [next_inputs["attention_mask"], torch.tensor([[1]])],
            dim=1),
    }

    next_token = tokenizer.decode(next_token_id)
    generated_tokens.append(next_token)

print(f"Total generation time: {sum(duration_s):.4f} seconds")
print(f"Generated tokens: {generated_tokens}")
