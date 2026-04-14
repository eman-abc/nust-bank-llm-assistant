import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

model_id = "eman-abc/gemma-3-4b-bank-merged"

def run_test():
    print(f"--- 1. Starting Local Load for: {model_id} ---")
    start_time = time.time()
    
    try:
        # We use bfloat16 to keep the size at ~8GB. 
        # Float32 would take 16GB which might crash your RAM.
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=torch.bfloat16, 
            device_map="cpu", # Force CPU
            trust_remote_code=True
        )
        
        load_time = time.time() - start_time
        print(f"--- 2. Model Loaded in {load_time:.2f} seconds ---")
        
        input_text = "What are the requirements for a Roshan Digital Account?"
        print(f"\nQUERY: {input_text}\n")
        
        inputs = tokenizer(input_text, return_tensors="pt")
        
        print("--- 3. Generating (this will be slow on CPU)... ---")
        gen_start = time.time()
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=50, 
                do_sample=True, 
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
            
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        gen_time = time.time() - gen_start
        
        print("\n--- FINAL RESPONSE ---")
        print(response)
        print(f"\n--- Generation took {gen_time:.2f} seconds ---")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("Hint: If it says 'Out of Memory', you might need more System RAM.")

if __name__ == "__main__":
    run_test()
