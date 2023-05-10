# Personal LLM Assistant Project (2023)

A lightweight language model fine-tuning project that trains a small LLM to answer questions about a person based on their profile information.

## Overview

This project demonstrates how to fine-tune a small language model (GPT-2) on personal data to create a personalized assistant that can answer questions about an individual. The model is designed to run locally on a standard laptop with moderate specifications (i5 processor, 8GB RAM).

## Requirements

- Python 3.7+
- PyTorch 1.11.0
- Transformers 4.18.0
- CUDA-compatible GPU (optional but recommended)

## Setup

1. Clone this repository
2. Create a virtual environment and install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate
pip install torch==1.11.0 transformers==4.18.0 tqdm
```

## Data Preparation

Place a text file containing the personal information in the `data` directory. For example:
```
data/profile.txt
```

## Usage

### Training the Model

To fine-tune the model on your personal data:

```bash
python personal_llm.py train --model_name=gpt2 --input_file=data/profile.txt --output_dir=model_output --num_epochs=20
```

Parameters:
- `model_name`: Base model to use (default: "gpt2")
- `input_file`: Path to your personal text data
- `output_dir`: Directory to save the fine-tuned model
- `batch_size`: Batch size for training (default: 4)
- `num_epochs`: Number of training epochs (default: 20)
- `learning_rate`: Learning rate (default: 5e-5)
- `block_size`: Block size for training (default: 128)

### Generating Responses

To interact with your trained model:

```bash
python personal_llm.py generate --model_dir=model_output
```

Parameters:
- `model_dir`: Path to the fine-tuned model
- `max_length`: Maximum length of generated text (default: 100)
- `temperature`: Temperature for sampling (default: 0.7)
- `top_k`: Top-k sampling parameter (default: 50)
- `top_p`: Top-p sampling parameter (default: 0.95)

## Performance Notes

- The model works best with a laptop having at least 8GB RAM.
- Fine-tuning is accelerated with CUDA-compatible GPUs.
- For systems with limited resources, reduce batch size and use a smaller version of GPT-2.

## Limitations

- The model is fine-tuned on very limited personal data, so responses may not always be accurate.
- The model may generate fictional information when asked about topics not covered in the training data.
- Performance is limited by the base model's capabilities (GPT-2 small in this case).

## Future Improvements

- Implement better prompt engineering for more reliable responses
- Add data augmentation to improve model performance with minimal data
- Implement quantization for even better performance on low-resource hardware

## License

MIT License