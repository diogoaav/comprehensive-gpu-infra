from datasets import load_from_disk
from transformers import AutoTokenizer
import argparse
import os
from concurrent.futures import ProcessPoolExecutor
import numpy as np

def tokenize_function(examples, tokenizer):
    return tokenizer(examples['text'], 
                    truncation=True, 
                    max_length=512, 
                    padding='max_length')

def process_chunk(args):
    chunk_id, chunk_data, tokenizer_name, output_dir = args
    
    # Initialize tokenizer for this process
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Tokenize the chunk
    tokenized_data = tokenize_function(chunk_data, tokenizer)
    
    # Save the chunk
    output_path = os.path.join(output_dir, f'chunk_{chunk_id:05d}.npy')
    np.savez(output_path, 
             input_ids=tokenized_data['input_ids'],
             attention_mask=tokenized_data['attention_mask'])
    
    return chunk_id

def main():
    parser = argparse.ArgumentParser(description='Tokenize OpenWebText dataset')
    parser.add_argument('--input-dir', type=str, required=True,
                       help='Directory containing the dataset')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Directory to save tokenized data')
    parser.add_argument('--tokenizer', type=str, default='gpt2',
                       help='Tokenizer to use (default: gpt2)')
    parser.add_argument('--num-workers', type=int, default=8,
                       help='Number of parallel workers')
    parser.add_argument('--chunk-size', type=int, default=1000,
                       help='Number of examples per chunk')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load dataset
    print(f"Loading dataset from {args.input_dir}")
    dataset = load_from_disk(args.input_dir)
    
    # Prepare chunks for parallel processing
    num_examples = len(dataset)
    chunk_indices = range(0, num_examples, args.chunk_size)
    chunks = [(i//args.chunk_size, 
               dataset[i:i+args.chunk_size], 
               args.tokenizer, 
               args.output_dir) 
              for i in chunk_indices]
    
    print(f"Processing {len(chunks)} chunks with {args.num_workers} workers")
    
    # Process chunks in parallel
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        for chunk_id in executor.map(process_chunk, chunks):
            print(f"Completed chunk {chunk_id}")
    
    print("Tokenization complete!")

if __name__ == "__main__":
    main() 