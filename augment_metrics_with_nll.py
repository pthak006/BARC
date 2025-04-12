#!/usr/bin/env python
# coding=utf-8

"""
Augments an existing metrics CSV file with NLL/bit-length calculations
using multiple specified Code LLM models sequentially.

Reads prompts and responses from the original JSONL answer file and
existing metrics from the input CSV file. Outputs an augmented CSV,
saving the state after each model completes processing.
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
        if input_ids.shape[-1] >= model_max_len:
             logger.warning(f"Input + Completion ({input_ids.shape[-1]} tokens) exceeds model max length ({model_max_len}). Skipping NLL.")
             return None

        # Re-tokenize prompt to get its length *with special tokens*
        prompt_encoding_for_len = tokenizer(prompt_text, add_special_tokens=True, return_tensors='pt', max_length=model_max_len, truncation=True) # Add truncation just in case
        prompt_tokens_length = prompt_encoding_for_len['input_ids'].shape[1]

        if prompt_tokens_length >= input_ids.shape[-1]:
             logger.warning(f"Prompt length ({prompt_tokens_length}) >= Full text length ({input_ids.shape[-1]}) after tokenization. Skipping NLL.")
             return None

        labels = input_ids.clone()
        labels[:, :prompt_tokens_length] = -100
        num_predicted_tokens = (labels != -100).sum().item()
        if num_predicted_tokens == 0:
             logger.warning("No tokens to predict after masking prompt. Skipping NLL.")
             return None

        input_ids = input_ids.to(device)
        labels = labels.to(device)
        outputs = model(input_ids=input_ids, labels=labels)
        avg_nll = outputs.loss.item() # This is already the average NLL
        if not math.isfinite(avg_nll):
             logger.warning("Non-finite average NLL received from model. Skipping NLL.")
             return None

        # The loss returned by HF models with labels is the *average* cross-entropy loss
        # over the non-masked tokens.
        # Total NLL = Average NLL * Number of Predicted Tokens
        total_nll = avg_nll * num_predicted_tokens
        return total_nll
    except Exception as e:
        logger.error(f"Error during NLL calculation: {e}", exc_info=False) # Set exc_info=True for full traceback if needed
        return None
# --- End NLL Calculation ---

# --- Main Augmentation Function ---
def main():
    parser = argparse.ArgumentParser(description="Augment metrics CSV with NLL from multiple models sequentially, saving after each model.")
    parser.add_argument("--answer_file", required=True, help="Path to the ORIGINAL .jsonl inference file ('uid', 'prompt', 'responses').")
    parser.add_argument("--existing_metrics_csv", required=True, help="Path to the existing metrics CSV file (e.g., 'results/metrics.csv'). This file will be overwritten after each model.")
    parser.add_argument("--output_augmented_csv", required=True, help="Path to save the augmented metrics CSV file. This file will be overwritten after each model.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device for NLL models ('cuda' or 'cpu').")
    # Optional: limit responses for testing (REMOVED as per original code, but could be added back)
    # parser.add_argument("--max_responses_per_problem", type=int, default=None, help="Maximum number of responses to process per problem (for debugging/testing).")

    args = parser.parse_args()

    # --- Define Models to Process ---
    # Assumes 7B NLL might already be in the existing_metrics_csv or will be added if listed here
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
    logger.info(f"Will calculate NLL sequentially using models: {nll_model_ids}")
    logger.info(f"Output will be saved to: {args.output_augmented_csv} after each model.")

    # --- Load Existing Metrics ---
    if not os.path.exists(args.existing_metrics_csv):
        logger.error(f"Existing metrics file not found: {args.existing_metrics_csv}")
        sys.exit(1)
    logger.info(f"Loading initial metrics from: {args.existing_metrics_csv}")
    try:
        df_metrics = pd.read_csv(args.existing_metrics_csv)
        # Ensure essential columns exist from previous run
        required_cols = ['uid', 'response_index', 'is_correct', 'raw_length', 'code_parse_error'] # Add any other essential columns
        if not all(col in df_metrics.columns for col in required_cols):
             missing_cols = [col for col in required_cols if col not in df_metrics.columns]
             logger.error(f"Existing metrics CSV missing required columns: {missing_cols}")
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


    # --- Iterate Through Models and Calculate NLL ---
    for model_id in nll_model_ids:
        model_column_name = nll_column_names[model_id]
        logger.info(f"\n--- Processing Model: {model_id} ---")
        logger.info(f"--- Target Column: {model_column_name} ---")

        # --- Add Column JUST for this model if needed ---
        # Check if the column already exists from a previous run or earlier model
        if model_column_name not in df_metrics.columns:
            df_metrics[model_column_name] = np.nan
            logger.info(f"Added column: {model_column_name}")
        else:
            logger.warning(f"Column {model_column_name} already exists. Existing values may be overwritten if calculation is repeated.")
            # Optional: Add logic here to skip calculation if column is fully populated
            # if df_metrics[model_column_name].notna().all():
            #    logger.info(f"Column {model_column_name} seems fully populated. Skipping NLL calculation for this model.")
            #    continue

        # --- Load Current NLL Model ---
        nll_model = None
        nll_tokenizer = None
        nll_model_max_len = None
        try:
            logger.info(f"Loading NLL model/tokenizer: {model_id}")
            # Set explicit cache dir if needed: cache_dir="..."
            nll_tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            # device_map='auto' is generally preferred for multi-GPU or large models.
            # If using a single GPU that fits the model, device_map=args.device is fine.
            nll_model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=torch.bfloat16, # Use bfloat16 for efficiency if supported
                device_map='auto', # Changed to 'auto' for better flexibility
                trust_remote_code=True,
                low_cpu_mem_usage=True # Useful for large models
            ).eval()
            nll_model_max_len = getattr(nll_model.config, 'max_position_embeddings', None)
            if nll_model_max_len is None:
                 logger.warning(f"Could not determine max_position_embeddings for {model_id}. Using default 2048.")
                 nll_model_max_len = 2048 # Fallback if not found

            # Check model device placement if device_map='auto' was used
            try:
                logger.info(f"Model device map: {nll_model.hf_device_map}")
                # Determine primary device for placing tensors if needed later (usually handled by device_map)
                primary_device = next(iter(set(nll_model.hf_device_map.values())))
                logger.info(f"Primary device detected as: {primary_device}")
            except AttributeError:
                 logger.info(f"Model loaded directly to device: {args.device}") # Fallback if hf_device_map not present
                 primary_device = args.device

            logger.info(f"Model loaded. Max length: {nll_model_max_len}")

        except Exception as e:
            logger.error(f"FAILED to load model {model_id}: {e}. Skipping this model.", exc_info=True)
            continue # Skip to the next model

        # --- Calculate NLL for all applicable rows for THIS model ---
        processed_count = 0
        skipped_existing = 0
        skipped_parse_error = 0
        skipped_lookup_error = 0
        skipped_nll_error = 0

        # Use tqdm on the DataFrame iterator
        # We iterate through all rows each time, but only calculate if needed.
        for index, row in tqdm(df_metrics.iterrows(), total=len(df_metrics), desc=f"Calculating NLL ({model_id.split('/')[-1]})"):

            # 1. Skip if code parsing failed previously
            if row['code_parse_error']:
                skipped_parse_error += 1
                continue

            # 2. Skip if NLL already calculated for THIS model in a previous run (allows resuming)
            if pd.notna(row[model_column_name]):
                skipped_existing += 1
                continue

            # --- Retrieve original prompt and response text ---
            uid = row['uid']
            response_idx = int(row['response_index']) # Ensure index is int for lookup

            if uid in answers_data and response_idx in answers_data[uid]:
                original_prompt = answers_data[uid][response_idx]['prompt']
                response_text = answers_data[uid][response_idx]['response']
            else:
                logger.warning(f"Could not find original prompt/response for {uid}, index {response_idx}. Skipping NLL for this entry.")
                skipped_lookup_error += 1
                # Optionally mark this row with an error state in the NLL column?
                # df_metrics.loc[index, model_column_name] = -1 # Or some other indicator
                continue

            # --- Parse the code ---
            # parse_code returns a LIST of strings or potentially an empty list
            parsed_codes_list = parse_code(response_text)

            # Check if parsing was successful AND the list is not empty
            # This check might be redundant if 'code_parse_error' is reliable, but good as a safeguard
            if not parsed_codes_list:
                logger.warning(f"Code parsing returned empty list for {uid}, index {response_idx}, though code_parse_error flag was False. Skipping NLL.")
                # Ensure the parse error flag is set if it wasn't before
                if not df_metrics.loc[index, 'code_parse_error']:
                     df_metrics.loc[index, 'code_parse_error'] = True # Correct the flag if needed
                skipped_parse_error += 1
                continue

            # Extract the first code string from the list
            # Assuming we only care about the first valid code block found
            code_string = parsed_codes_list[0]

            # --- Calculate NLL using the extracted code STRING ---
            # Pass the primary device determined after loading the model
            total_nll = calculate_nll(original_prompt, code_string, nll_model, nll_tokenizer, primary_device, nll_model_max_len)

            # --- Store Bit Length ---
            if total_nll is not None and math.isfinite(total_nll):
                bit_length = total_nll / math.log(2)
                df_metrics.loc[index, model_column_name] = bit_length
                processed_count += 1
            else:
                 # NLL calculation failed or returned non-finite value
                 logger.warning(f"NLL calculation failed or returned invalid value for {uid}, index {response_idx}.")
                 skipped_nll_error += 1
                 # Optionally mark with NaN (already default) or a specific error code if needed

        # --- Log summary for this model ---
        logger.info(f"Finished NLL calculation for {model_id}.")
        logger.info(f"  Successfully calculated NLL for: {processed_count} solutions.")
        logger.info(f"  Skipped (already had value):    {skipped_existing} solutions.")
        logger.info(f"  Skipped (code parse error):     {skipped_parse_error} solutions.")
        logger.info(f"  Skipped (lookup error):         {skipped_lookup_error} solutions.")
        logger.info(f"  Skipped (NLL calculation error):{skipped_nll_error} solutions.")
        logger.info(f"  Total rows processed/skipped:   {processed_count + skipped_existing + skipped_parse_error + skipped_lookup_error + skipped_nll_error} / {len(df_metrics)}")


        # --- Save Intermediate State After Processing This Model ---
        logger.info(f"Saving augmented metrics (state after {model_id}) to: {args.output_augmented_csv}")
        try:
            # Ensure output directory exists (important on first save)
            output_dir = os.path.dirname(args.output_augmented_csv)
            if output_dir and not os.path.exists(output_dir):
                 os.makedirs(output_dir, exist_ok=True)
                 logger.info(f"Created output directory: {output_dir}")
            df_metrics.to_csv(args.output_augmented_csv, index=False)
            logger.info(f"Metrics including {model_column_name} saved successfully.")
        except Exception as e:
             logger.error(f"Failed to save intermediate metrics CSV after {model_id}: {e}", exc_info=True)
             # Decide if we should continue or stop? For now, log and continue.
             # Consider adding a flag or sys.exit(1) if saving is critical between steps.


        # --- Unload Model ---
        logger.info(f"Unloading model: {model_id}")
        try:
            del nll_model
            del nll_tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Model unloaded and CUDA cache cleared (if applicable).")
        except Exception as e:
             logger.error(f"Error unloading model {model_id}: {e}")
        # End of outer model loop

    logger.info("\n--- All models processed. Final augmented metrics saved. ---")


if __name__ == "__main__":
    main()