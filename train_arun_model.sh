#!/bin/bash

# Create directories
mkdir -p data model_output logs

# Copy profile.txt to data directory if it doesn't exist
if [ ! -f "data/profile.txt" ]; then
    cp profile.txt data/profile.txt
    echo "Copied profile.txt to data directory"
fi

# Check if virtual environment exists, create if not
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python -m venv .venv
    source .venv/bin/activate
    pip install torch==1.11.0 transformers==4.18.0 tqdm colorama
else
    source .venv/bin/activate
fi

# Train the model
echo "Starting model training..."
python personal_llm.py train \
    --model_name=gpt2 \
    --input_file=data/profile.txt \
    --output_dir=model_output \
    --batch_size=2 \
    --num_epochs=30 \
    --learning_rate=5e-5 \
    --block_size=64 \
    2>&1 | tee logs/training_log.txt

echo "Training completed. Model saved to model_output/"
echo "You can now chat with the model using: python chat_with_arun.py --model_dir=model_output"