#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
# WARNING: Storing tokens directly in scripts is a security risk.
# Consider using environment variables or a secret management tool for production.
HUGGINGFACE_TOKEN="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"
WANDB_TOKEN="3189cdd95c63d2408e95da0b79a5439d79f145a5"

# --- Installation Steps ---
echo "Installing PyTorch..."
pip install torch==2.4

echo "Installing flash-attn..."
pip install flash-attn --no-build-isolation

echo "Installing alignment-handbook requirements..."
# Navigate into the alignment-handbook directory
cd finetune/alignment-handbook/
# Install the package located in the current directory (.)
python -m pip install .
# Navigate back to the original root directory
cd ../../

echo "Installing main requirements..."
pip install -r requirements.txt

echo "Installing vLLM..."
# Pinned version as requested
pip install vllm==0.6.0

echo "Installing Pebble..."
pip install pebble

# --- Logins ---
echo "Logging into Hugging Face CLI..."
# Use the token and explicitly answer 'no' (via False) to adding it as a git credential
huggingface-cli login --token "$HUGGINGFACE_TOKEN" --add-to-git-credential False

echo "Logging into Wandb..."
# Provide the token directly to the login command for non-interactive login
wandb login "$WANDB_TOKEN"

# --- Environment Variable ---
echo "Setting TOKENIZERS_PARALLELISM..."
# This export command sets the variable for the duration of this script execution.
# If you need this variable set in your shell *after* this script finishes,
# you will need to either:
# 1. Run `export TOKENIZERS_PARALLELISM=false` manually in your terminal, OR
# 2. Source this script instead of running it directly: `source setup_environment.sh`
export TOKENIZERS_PARALLELISM=false

echo "Setup complete!"
echo "Note: TOKENIZERS_PARALLELISM is set to false for this script's execution."
echo "Run 'export TOKENIZERS_PARALLELISM=false' or source this script ('source setup_environment.sh') if you need it in your current shell session afterwards."
