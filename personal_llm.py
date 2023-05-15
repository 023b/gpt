import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import logging
import argparse

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

class PersonalTextDataset(Dataset):
    def __init__(self, tokenizer, file_path, block_size=128):
        logger.info(f"Reading data from {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        self.examples = []
        tokenized_text = tokenizer.encode(text)
        
        for i in range(0, len(tokenized_text) - block_size, block_size // 2):
            self.examples.append(tokenized_text[i:i + block_size])
        
        logger.info(f"Created {len(self.examples)} training examples from the input text")
        
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return torch.tensor(self.examples[idx])

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load pre-trained model and tokenizer
    logger.info(f"Loading model: {args.model_name}")
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_name)
    model = GPT2LMHeadModel.from_pretrained(args.model_name)
    model.to(device)
    
    # Create dataset
    dataset = PersonalTextDataset(tokenizer, args.input_file, args.block_size)
    
    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Prepare optimizer and scheduler
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    
    # Calculate total training steps
    total_steps = len(dataloader) * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps
    )
    
    # Train the model
    logger.info("Starting training...")
    model.train()
    
    for epoch in range(args.num_epochs):
        epoch_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs}"):
            batch = batch.to(device)
            
            outputs = model(batch, labels=batch)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            epoch_loss += loss.item()
        
        avg_epoch_loss = epoch_loss / len(dataloader)
        logger.info(f"Epoch {epoch+1}/{args.num_epochs} - Average Loss: {avg_epoch_loss:.4f}")
        
        # Save model checkpoint
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{epoch+1}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)
        logger.info(f"Saved model checkpoint to {checkpoint_dir}")
    
    # Save final model
    logger.info("Training completed. Saving final model...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info(f"Final model saved to {args.output_dir}")

def generate_response(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load fine-tuned model and tokenizer
    logger.info(f"Loading model from {args.model_dir}")
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_dir)
    model = GPT2LMHeadModel.from_pretrained(args.model_dir)
    model.to(device)
    model.eval()
    
    while True:
        query = input("Ask a question about Arun (or type 'exit' to quit): ")
        if query.lower() == 'exit':
            break
        
        # Prepare prompt
        prompt = f"Question: {query}\nAnswer:"
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        
        # Generate response
        output = model.generate(
            input_ids,
            max_length=args.max_length + len(input_ids[0]),
            num_return_sequences=1,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        response = response.replace(prompt, "").strip()
        print(f"\nResponse: {response}\n")

def main():
    parser = argparse.ArgumentParser(description="Fine-tune a language model on personal data")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--model_name", type=str, default="gpt2", help="Model name or path")
    train_parser.add_argument("--input_file", type=str, required=True, help="Path to input text file")
    train_parser.add_argument("--output_dir", type=str, required=True, help="Output directory for model")
    train_parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    train_parser.add_argument("--num_epochs", type=int, default=20, help="Number of training epochs")
    train_parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    train_parser.add_argument("--block_size", type=int, default=128, help="Block size for training")
    
    # Generate command
    generate_parser = subparsers.add_parser("generate", help="Generate responses")
    generate_parser.add_argument("--model_dir", type=str, required=True, help="Path to fine-tuned model")
    generate_parser.add_argument("--max_length", type=int, default=100, help="Maximum length of generated text")
    generate_parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for sampling")
    generate_parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling parameter")
    generate_parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling parameter")
    
    args = parser.parse_args()
    
    if args.command == "train":
        train(args)
    elif args.command == "generate":
        generate_response(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()