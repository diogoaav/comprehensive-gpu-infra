from datasets import load_dataset
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description='Load OpenWebText dataset')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Directory to save the dataset')
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Load the dataset
    print("Loading OpenWebText dataset...")
    dataset = load_dataset("openwebtext", split="train")
    
    # Save the dataset
    print(f"Saving dataset to {args.output_dir}")
    dataset.save_to_disk(args.output_dir)
    
    print("Dataset loading complete!")

if __name__ == "__main__":
    main() 