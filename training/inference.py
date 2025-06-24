import torch
import os
from train_gpt import GPT, GPTConfig
import tiktoken

def load_model(checkpoint_path):
    """Load a trained model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config = checkpoint['config']
    
    model = GPT(config)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    return model, config

def generate_text(model, config, prompt, max_new_tokens=100, temperature=0.8, top_k=50):
    """Generate text from a prompt"""
    # Initialize tokenizer (GPT-2 tokenizer)
    enc = tiktoken.get_encoding("gpt2")
    
    # Encode prompt
    tokens = enc.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)  # Add batch dimension
    
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Crop tokens if sequence is too long
            if tokens.size(1) > config.block_size:
                tokens = tokens[:, -config.block_size:]
            
            # Forward pass
            logits, _ = model(tokens)
            logits = logits[:, -1, :] / temperature  # Get last token logits and apply temperature
            
            # Apply top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Sample from the distribution
            probs = torch.nn.functional.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            tokens = torch.cat((tokens, next_token), dim=1)
    
    # Decode and return
    generated_tokens = tokens[0].tolist()
    generated_text = enc.decode(generated_tokens)
    
    return generated_text

def main():
    checkpoint_path = "/mnt/jfs/checkpoints/final_model.pt"
    
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Available checkpoints:")
        checkpoint_dir = "/mnt/jfs/checkpoints"
        if os.path.exists(checkpoint_dir):
            for f in os.listdir(checkpoint_dir):
                if f.endswith('.pt'):
                    print(f"  {f}")
        return
    
    print("Loading model...")
    model, config = load_model(checkpoint_path)
    
    print(f"Model loaded! Parameters: {model.get_num_params()/1e6:.2f}M")
    
    # Interactive text generation
    while True:
        prompt = input("\nEnter a prompt (or 'quit' to exit): ")
        if prompt.lower() == 'quit':
            break
        
        print("Generating...")
        generated = generate_text(model, config, prompt, max_new_tokens=50)
        print(f"\nGenerated text:\n{generated}")

if __name__ == "__main__":
    main() 