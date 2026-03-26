import torch
import re
import argparse
import pandas as pd
from collections import Counter

# Import classes from your training script
from Main import TextPreprocessor, Tokenizer, ResearchTopicGPT  # Adjust import if needed

def load_model(model_path, tokenizer, device='cpu'):
    """Load a trained model with the correct architecture"""
    # Initialize model with the same hyperparameters as training
    model = ResearchTopicGPT(
        vocab_size=len(tokenizer.word2idx),
        d_model=384,
        nhead=8,
        num_layers=8,
        max_length=100
    )
    
    # Load trained weights
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    model.eval()
    return model

def generate_topics(model, tokenizer, start_words, num_topics=5, temperature=0.8, top_p=0.9):
    """Generate research topics from starting words"""
    topics = []
    for word in start_words:
        for _ in range(num_topics):
            topic = model.generate(tokenizer, word, temperature=temperature, top_p=top_p)
            topics.append(f"{word}: {topic}")
    return topics

def main():
    parser = argparse.ArgumentParser(description='Generate research topics from a trained model.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model weights')
    parser.add_argument('--data_path', type=str, required=True, help='Path to original CSV data')
    parser.add_argument('--start_words', nargs='+', required=True, help='Starting words for generation')
    parser.add_argument('--num_topics', type=int, default=5, help='Number of topics to generate per word')
    parser.add_argument('--temperature', type=float, default=0.8, help='Creativity control (0.5-1.5)')
    parser.add_argument('--top_p', type=float, default=0.9, help='Diversity control (0.7-1.0)')
    parser.add_argument('--output', type=str, default='generated_topics.txt', help='Output file')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Recreate tokenizer from original data
    print("Preprocessing data and building tokenizer...")
    preprocessor = TextPreprocessor()
    topics = preprocessor.process_csv(args.data_path)
    tokenizer = Tokenizer(topics)
    
    # Load trained model
    print(f"Loading model from {args.model_path}...")
    model = load_model(args.model_path, tokenizer, device)
    model.to(device)
    
    # Generate topics
    print(f"Generating {args.num_topics} topics per starting word...")
    generated = generate_topics(
        model, 
        tokenizer,
        args.start_words,
        num_topics=args.num_topics,
        temperature=args.temperature,
        top_p=args.top_p
    )
    
    # Save results
    with open(args.output, 'w') as f:
        f.write("Generated Research Topics:\n")
        f.write("==========================\n\n")
        for i, topic in enumerate(generated, 1):
            f.write(f"{i}. {topic}\n")
    
    print(f"Successfully generated {len(generated)} topics saved to {args.output}")
    print("\nSample topics:")
    for i, topic in enumerate(generated[:5], 1):
        print(f"{i}. {topic}")

if __name__ == "__main__":
    main()