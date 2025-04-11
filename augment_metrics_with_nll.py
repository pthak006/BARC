#!/usr/bin/env python
# coding=utf-8

"""
Augments an existing metrics CSV file with NLL/bit-length calculations
using multiple specified Code LLM models sequentially.

Reads prompts and responses from the original JSONL answer file and
existing metrics from the input CSV file. Outputs an augmented CSV.
"""

import os
import sys
import json
import argparse
import math
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import logging
from typing import Optional, Dict, List

# --- Imports from BARC ---
# Only need parse_code from BARC utils now
try:
    from utils import parse_code
except ImportError:
    print("ERROR: Could not import 'parse_code' from 'utils'. Ensure BARC utils are in PYTHONPATH.")
    sys.exit(1)
# --- End BARC Imports ---

# --- Logging Setup ---
LOG_FILENAME = 'augment_metrics.log' # New log file name
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] %(message)s',
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILENAME, mode='w'), # Overwrite log file each run
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)
logger.info(f"Logging to console and file: {LOG_FILENAME}")
# --- End Logging Setup ---

# --- NLL Calculation Function (Keep as before) ---
@torch.no_grad()
def calculate_nll(prompt_text: str, completion_text: str, model, tokenizer, device='cuda', model_max_len=None) -> Optional[float]:
    """Calculates the total NLL of the completion given the prompt."""
    if model_max_len is None:
        model_max_len = getattr(model.config, 'max_position_embeddings', 2048)
    try:
        full_text = prompt_text + completion_text
        # Using add_special_tokens=True here assuming model expects BOS/EOS for likelihood calc
        encoding = tokenizer(full_text, add_special_tokens=True, return_tensors='pt', truncation=True, max_length=model_max_len)
        input_ids = encoding['input_ids']
        if input_ids.shape[-1] >= model_max_len: return None
        # Re-tokenize prompt to get its length *with special tokens*
        prompt_encoding_for_len = tokenizer(prompt_text, add_special_tokens=True, return_tensors='pt')
        prompt_tokens_length = prompt_encoding_for_len['input_ids'].shape[1]
        if prompt_tokens_length >= input_ids.shape[-1]: return None
        labels = input_ids.clone()
        labels[:, :prompt_tokens_length] = -100
        num_predicted_tokens = (labels != -100).sum().item()
        if num_predicted_tokens == 0: return None
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        outputs = model(input_ids=input_ids, labels=labels)
        avg_nll = outputs.loss.item()
        if not math.isfinite(avg_nll): return None
        total_nll = avg_nll * num_predicted_tokens
        return total_nll
    except Exception as e:
        logger.error(f"Error during NLL calculation: {e}", exc_info=False)
        return None
# --- End NLL Calculation ---

# --- Main Augmentation Function ---
def main():
    parser = argparse.ArgumentParser(description="Augment metrics CSV with NLL from multiple models.")
    parser.add_argument("--answer_file", required=True, help="Path to the ORIGINAL .jsonl inference file ('uid', 'prompt', 'responses').")
    parser.add_argument("--existing_metrics_csv", required=True, help="Path to the existing metrics CSV file (e.g., 'results/metrics.csv').")
    parser.add_argument("--output_augmented_csv", required=True, help="Path to save the new augmented metrics CSV file.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device for NLL models ('cuda' or 'cpu').")
    # Optional: limit responses for testing
    parser.add_argument("--max_responses_per_problem", type=int, default=None, help="Maximum number of responses to process per problem (for debugging/testing).")

    args = parser.parse_args()

    # --- Define Models to Process ---
    # Assumes 7B NLL is already in the existing_metrics_csv
    nll_model_ids = [
        "codellama/CodeLlama-13b-Python-hf",
        "codellama/CodeLlama-34b-Python-hf",
        "codellama/CodeLlama-70b-Python-hf",
        # Add or remove models as needed
    ]
    # Create corresponding column names
    nll_column_names = {
        model_id: f"bit_length_{model_id.split('/')[-1].replace('-Python-hf', '').replace('CodeLlama-', '')}"
        for model_id in nll_model_ids
    }
    logger.info(f"Will calculate NLL using models: {nll_model_ids}")

    # --- Load Existing Metrics ---
    if not os.path.exists(args.existing_metrics_csv):
        logger.error(f"Existing metrics file not found: {args.existing_metrics_csv}")
        sys.exit(1)
    logger.info(f"Loading existing metrics from: {args.existing_metrics_csv}")
    try:
        df_metrics = pd.read_csv(args.existing_metrics_csv)
        # Ensure essential columns exist from previous run
        required_cols = ['uid', 'response_index', 'is_correct', 'raw_length', 'code_parse_error']
        if not all(col in df_metrics.columns for col in required_cols):
             logger.error(f"Existing metrics CSV missing required columns: {required_cols}")
             sys.exit(1)
        logger.info(f"Loaded {len(df_metrics)} rows from existing metrics CSV.")
    except Exception as e:
        logger.error(f"Failed to load existing metrics CSV: {e}", exc_info=True)
        sys.exit(1)

    # --- Load Original Answers (for prompts and code text) ---
    if not os.path.exists(args.answer_file):
        logger.error(f"Original answer file not found: {args.answer_file}")
        sys.exit(1)
    logger.info(f"Loading original prompts/responses from: {args.answer_file}")
    answers_data = {} # Store as uid -> {response_idx -> {'prompt': ..., 'response': ...}}
    try:
        with open(args.answer_file) as f: lines = f.readlines()
        problem_answers = [json.loads(line) for line in lines]
        for p in problem_answers:
            uid = p.get("uid")
            prompt = p.get("prompt")
            responses = p.get("responses", [])
            if uid and prompt: # Need uid and prompt
                answers_data[uid] = {
                    idx: {'prompt': prompt, 'response': resp_text}
                    for idx, resp_text in enumerate(responses)
                }
    except Exception as e:
        logger.error(f"Failed to load or structure original answers JSONL: {e}", exc_info=True)
        sys.exit(1)
    logger.info(f"Structured original answers for {len(answers_data)} problems.")


    # --- Add New Columns to DataFrame ---
    for model_id in nll_model_ids:
        col_name = nll_column_names[model_id]
        if col_name not in df_metrics.columns:
            df_metrics[col_name] = np.nan
            logger.info(f"Added column: {col_name}")
        else:
            logger.info(f"Column {col_name} already exists. Will overwrite.")

    # --- Iterate Through Models and Calculate NLL ---
    for model_id in nll_model_ids:
        model_column_name = nll_column_names[model_id]
        logger.info(f"\n--- Processing Model: {model_id} ---")
        logger.info(f"--- Target Column: {model_column_name} ---")

        # --- Load Current NLL Model ---
        nll_model = None
        nll_tokenizer = None
        nll_model_max_len = None
        try:
            logger.info(f"Loading NLL model/tokenizer: {model_id}")
            # Set explicit cache dir if needed: cache_dir="..."
            nll_tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            # device_map=args.device might cause issues with large models on single GPU if not enough RAM.
            # Consider device_map='auto' or manually placing layers if needed, but start with simple assignment.
            nll_model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=torch.bfloat16,
                device_map=args.device, trust_remote_code=True
                # Add low_cpu_mem_usage=True if needed for large models
            ).eval()
            nll_model_max_len = getattr(nll_model.config, 'max_position_embeddings', None)
            logger.info(f"Model loaded. Max length: {nll_model_max_len}")
        except Exception as e:
            logger.error(f"FAILED to load model {model_id}: {e}. Skipping this model.", exc_info=True)
            continue # Skip to the next model

        # --- Calculate NLL for all applicable rows ---
        processed_count = 0
        # Use tqdm on the DataFrame iterator
        for index, row in tqdm(df_metrics.iterrows(), total=len(df_metrics), desc=f"Calculating NLL ({model_id.split('/')[-1]})"):
            # Skip if code parsing failed previously or NLL already calculated
            if row['code_parse_error'] or pd.notna(row[model_column_name]):
                continue

            uid = row['uid']
            response_idx = int(row['response_index']) # Ensure index is int for lookup

            # Retrieve original prompt and response text
            if uid in answers_data and response_idx in answers_data[uid]:
                original_prompt = answers_data[uid][response_idx]['prompt']
                response_text = answers_data[uid][response_idx]['response']
            else:
                logger.warning(f"Could not find original prompt/response for {uid}, index {response_idx}. Skipping NLL.")
                continue

            # Parse code again (needed for NLL calc)
            code = parse_code(response_text)
            if not code:
                # This case should be covered by 'code_parse_error' flag, but double-check
                logger.warning(f"Code parsing failed again for {uid}, index {response_idx}, though flag was False?")
                continue

            # Calculate NLL
            total_nll = calculate_nll(original_prompt, code, nll_model, nll_tokenizer, args.device, nll_model_max_len)

            # Store Bit Length
            if total_nll is not None:
                bit_length = total_nll / math.log(2)
                df_metrics.loc[index, model_column_name] = bit_length
                processed_count += 1

        logger.info(f"Finished NLL calculation for {model_id}. Calculated for {processed_count} solutions.")

        # --- Unload Model ---
        logger.info(f"Unloading model: {model_id}")
        try:
            del nll_model
            del nll_tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Model unloaded and cache cleared.")
        except Exception as e:
             logger.error(f"Error unloading model {model_id}: {e}")
        # End of outer model loop

    # --- Save Augmented Results ---
    logger.info(f"Saving augmented metrics ({len(df_metrics)} total rows) to: {args.output_augmented_csv}")
    try:
        output_dir = os.path.dirname(args.output_augmented_csv)
        if output_dir:
             os.makedirs(output_dir, exist_ok=True)
        df_metrics.to_csv(args.output_augmented_csv, index=False)
        logger.info("Augmented metrics saved successfully.")
    except Exception as e:
         logger.error(f"Failed to save augmented metrics CSV: {e}", exc_info=True)


if __name__ == "__main__":
    # Clear GPU memory before starting the script
    if torch.cuda.is_available():
        # Explicitly delete any existing model or tokenizer references
        import gc
        
        # Find and delete any model or tokenizer objects
        for obj in gc.get_objects():
            try:
                if isinstance(obj, (AutoModelForCausalLM, AutoTokenizer)) or \
                   "model" in str(type(obj)).lower() or \
                   "tokenizer" in str(type(obj)).lower():
                    del obj
            except:
                pass
        
        # Force garbage collection to remove deleted objects
        gc.collect()
        
        # Clear CUDA cache
        torch.cuda.empty_cache()
        
        # Reset memory statistics
        for i in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(i)
            torch.cuda.reset_accumulated_memory_stats(i)
            
        logger.info("All model and tokenizer references deleted and GPU memory cleared before starting script")
    main()