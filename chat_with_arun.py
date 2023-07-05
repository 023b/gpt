import os
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import argparse
import logging
import time
from colorama import Fore, Style, init

# Initialize colorama
init()

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

class ArunChatbot:
    def __init__(self, model_dir, max_length=100, temperature=0.7, top_k=50, top_p=0.95):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load model and tokenizer
        logger.info(f"Loading model from {model_dir}")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
        self.model = GPT2LMHeadModel.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()
        
        # Generation parameters
        self.max_length = max_length
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        
        # Welcome message
        self.welcome_message = """
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║  Welcome to Arun's Personal Assistant!                   ║
║                                                          ║
║  This AI has been trained on Arun's profile and can      ║
║  answer questions about his:                             ║
║    - Background and education                            ║
║    - Interests in AI and computing                       ║
║    - Skills and learning goals                           ║
║    - Future aspirations                                  ║
║                                                          ║
║  Type 'exit' at any time to quit the chat.               ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
"""
    
    def print_welcome(self):
        print(Fore.CYAN + self.welcome_message + Style.RESET_ALL)
    
    def generate_response(self, query):
        # Prepare prompt
        prompt = f"Question: {query}\nAnswer:"
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        # Add thinking animation
        print(Fore.YELLOW + "Thinking", end="")
        for _ in range(3):
            time.sleep(0.3)
            print(".", end="", flush=True)
        print(Style.RESET_ALL)
        
        # Generate response
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_length=self.max_length + len(input_ids[0]),
                num_return_sequences=1,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        response = response.replace(prompt, "").strip()
        
        return response
    
    def chat(self):
        self.print_welcome()
        
        while True:
            query = input(Fore.GREEN + "\nYou: " + Style.RESET_ALL)
            if query.lower() in ["exit", "quit", "bye"]:
                print(Fore.CYAN + "\nArun's Assistant: Goodbye! Have a great day!" + Style.RESET_ALL)
                break
            
            response = self.generate_response(query)
            print(Fore.CYAN + "\nArun's Assistant: " + Style.RESET_ALL + response)

def main():
    parser = argparse.ArgumentParser(description="Chat with Arun's personal assistant")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to fine-tuned model")
    parser.add_argument("--max_length", type=int, default=100, help="Maximum length of generated text")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for sampling")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling parameter")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling parameter")
    
    args = parser.parse_args()
    
    chatbot = ArunChatbot(
        model_dir=args.model_dir,
        max_length=args.max_length,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p
    )
    
    chatbot.chat()

if __name__ == "__main__":
    main()