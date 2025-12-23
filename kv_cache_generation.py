"""
Example script demonstrating efficient token generation using KV Caching with GPT-2.

KV (Key-Value) caching stores the Key and Value matrices calculated in previous attention steps.
Using `past_key_values`, the model only needs to compute attention for the *new* token,
drastically reducing computation time from O(N^2) to O(N) for generation.
"""

import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class ChatGPT2:
    """
    A wrapper class for GPT-2 generation with KV caching support.
    """
    def __init__(self):
        self.model_name = "./models/gpt2"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        # print(self.model)

    def generate_token(self, inputs):
        """
        Generate a single token using the model, utilizing past key-values for efficiency.

        Args:
            inputs (dict): Dictionary containing 'input_ids', 'attention_mask', 
                           and optionally 'past_key_values'.

        Returns:
            tuple:
                - next_token_id (torch.Tensor): The predicted next token ID.
                - past_key_values (tuple): Updated KV cache to be passed to the next iteration.
        """
        # torch.no_grad() tells PyTorch not to track gradients (inference mode)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        logits = outputs.logits
        # outputs.past_key_values contains the K/V states for all layers
        past_key_values = outputs.past_key_values
        
        # We only need the logits for the last token position to predict the next one
        last_logits = logits[0, -1, :]
        next_token_id = last_logits.argmax()
        
        return next_token_id, past_key_values

if __name__ == "__main__":
    chat_gp2 = ChatGPT2()
    
    # A long prompt to demonstrate that subsequent generation is fast even with long context
    prompt = f'''
    Please create a summary of the below content
    
    ## content:
    
    The city of Metronia has been facing a complex traffic crisis for over a decade. Daily congestion on its major arteries — Oakway Boulevard, Central Express, and the Downtown Loop — has resulted in an average commuter delay of 45 minutes per day. In 2018, the city council launched the “MoveMetronia” initiative to modernize the infrastructure, integrating smart traffic lights, AI-powered routing, and expanded bike lanes. However, implementation lagged due to budget constraints and coordination challenges among departments.
    In 2022, a new transportation commissioner prioritized data-driven solutions. The team partnered with a local university to deploy sensors that collected over 500 TB of real-time traffic data. The analytics revealed that 60% of congestion occurred from short-distance car trips that could be replaced by public transport or cycling. The report also highlighted inequities — while downtown had abundant bus lines, the outer districts suffered from low service frequency.
    To address these findings, Metronia introduced “FlexBus,” a demand-responsive transport service operating through a mobile app. Within six months, daily ridership increased by 22%, and average commute times dropped by 12%. The city also initiated a public campaign, encouraging citizens to adopt greener modes of transportation. Despite early challenges such as app reliability and driver shortages, user satisfaction reached 85% by mid-2024.
    Experts believe Metronia’s case exemplifies how mid-sized cities can leverage AI and behavioral insights to achieve sustainable mobility. The ongoing phase involves integrating predictive maintenance for traffic lights and optimizing EV charging infrastructure to further reduce carbon emissions.
    '''
    
    inputs = chat_gp2.tokenizer(prompt, return_tensors="pt")
    
    # Initialize loop variables
    next_inputs = inputs
    generated_tokens = []
    duration_in_seconds = []
    
    print("Starting generation with KV cache...")
    
    for _ in range(100):
        time_t0 = time.time()
        
        # Generate next token and get updated cache
        next_token_id, past_key_values = chat_gp2.generate_token(next_inputs)
        
        duration_in_seconds.append(time.time() - time_t0)

        # IMPORTANT: Prepare inputs for the next step.
        # Instead of passing the FULL sequence again (as in the naive example),
        # we only pass the NEW token ('input_ids') and the accumulated 'past_key_values'.
        next_inputs = {
            "input_ids": next_token_id.reshape((1, 1)),
            "attention_mask": torch.cat(
                [next_inputs["attention_mask"], torch.tensor([[1]])],
                dim=1),
            "past_key_values": past_key_values, # Adding the cache here is the magic sauce
        }


        next_token = chat_gp2.tokenizer.decode(next_token_id)
        generated_tokens.append(next_token)
        
        # Optional: Stop generation if a period is reached (commented out)
        # if next_token == ".":
        #     break

    print(f"\nGenerated text: {''.join(generated_tokens)}")
    print(f"Total generation time: {sum(duration_in_seconds):.4f} seconds")