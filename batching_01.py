import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class Batching:
    """
    A class to handle batching and token generation using a pre-trained GPT-2 model.
    """
    def __init__(self):
        """
        Initialize the Batching class by loading the tokenizer and model.
        The model is expected to be located at "./models/gpt2".
        """
        self.model_name = "./models/gpt2"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)

    def generate_token(self, inputs):
        """
        Generate the next token based on the provided inputs using the language model.

        Args:
            inputs (dict): A dictionary containing 'input_ids' and 'attention_mask'
                           (and optionally 'past_key_values' for caching).

        Returns:
            tuple: A tuple containing:
                - next_token_id (torch.Tensor): The ID of the most likely next token.
                - past_key_values (tuple): The cached key-value states to speed up future predictions.
        """
        # torch.no_grad() tells PyTorch not to track gradients (because this is inference, not training)
        # This reduces memory usage and speeds up computations.
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # The model’s output, outputs, includes logits — the raw, unnormalized prediction scores for each possible next token.
        logits = outputs.logits
        
        # past_key_values contains the pre-computed key and value states for the attention mechanism.
        # This allows the model to reuse computation for previous tokens when generating the next one.
        past_key_values = outputs.past_key_values
        
        # We are interested in the logits for the last token in the sequence (position -1).
        # logits shape: (batch_size, sequence_length, vocab_size)
        # We take the 0-th batch item (since batch size is likely 1) and the last token.
        last_logits = logits[0, -1, :]
        
        # Greedy decoding: Select the token with the highest score (probability).
        next_token_id = last_logits.argmax()
        
        return next_token_id, past_key_values