#!/usr/bin/env python
# coding=utf-8

"""
Analyzes generated ARC solutions from a JSONL file.
Calculates accuracy (based on training pairs), raw length (characters),
and bit-length (NLL using a specified Code LLM).
Saves the results to a CSV file.

Based on the original eval_code_samples.py from the BARC repository.
"""

import os
import sys
import json
import argparse
import math
import numpy as np
import pandas as pd # For saving to CSV
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import logging

# --- Imports from BARC ---
# Assume these are importable from the BARC project structure
try:
    from utils import parse_code
except ImportError:
    print("ERROR: Could not import 'parse_code' from 'utils'. Ensure BARC utils are in PYTHONPATH.")
    sys.exit(1)
try:
    from execution import multi_execute_transformation
except ImportError:
    print("ERROR: Could not import 'multi_execute_transformation' from 'execution'. Ensure BARC execution module is in PYTHONPATH.")
    sys.exit(1)
try:
    # If eval_code_samples.py is in the same dir or python path
    # Note: We copy relevant functions/classes here to make this script self-contained,
    #       as importing directly from a script might be fragile.
    # --- Copied/Adapted from eval_code_samples.py ---
    from enum import Enum
    class GridComparisonResult(Enum):
        EQUAL = 0
        SHAPE_MISMATCH = 1
        CONTENT_MISMATCH = 2
        TYPE_MISMATCH = 3
        ERROR = 4
        NON_2D_ARRAY = 5

    def compare_grids(output_grid, expected_output_grid):
        """Checks if output_grid == expected_output_grid, returns (result, ratio_of_correct)."""
        if isinstance(output_grid, str): return GridComparisonResult.ERROR, 0.0
        if not isinstance(output_grid, np.ndarray): return GridComparisonResult.TYPE_MISMATCH, 0.0
        if len(output_grid.shape) != 2: return GridComparisonResult.NON_2D_ARRAY, 0.0
        if output_grid.shape != expected_output_grid.shape: return GridComparisonResult.SHAPE_MISMATCH, 0.0
        if np.array_equal(output_grid, expected_output_grid): return GridComparisonResult.EQUAL, 1.0
        ratio = np.sum(output_grid == expected_output_grid) / np.prod(expected_output_grid.shape)
        return GridComparisonResult.CONTENT_MISMATCH, ratio

    # Need access to ARC problem data
    from arc import train_problems, validation_problems, ArcProblem
    # Combine problems (or load selectively based on UIDs in answer_file if preferred)
    # Note: Loading ConceptARC problems might need adjustment if not needed here.
    all_arc_problems_dict = {p.uid: p for p in (train_problems + validation_problems)}
    def get_arc_problem(uid):
        problem = all_arc_problems_dict.get(uid)
        if problem is None:
             raise ValueError(f"Problem {uid} not found in preloaded train/validation sets.")
        return problem
    # --- End Copied/Adapted ---

except ImportError as e:
    print(f"ERROR: Could not import required components: {e}. Ensure BARC environment is set up.")
    sys.exit(1)
# --- End BARC Imports ---

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- NLL Calculation Function ---
@torch.no_grad()
def calculate_nll(prompt_text: str, completion_text: str, model, tokenizer, device='cuda', model_max_len=None) -> Optional[float]:
    """Calculates the total negative log-likelihood of the completion_text given the prompt_text."""
    # Use the model's configured max length if not provided
    if model_max_len is None:
        model_max_len = getattr(model.config, 'max_position_embeddings', 2048)

    try:
        # Tokenize separately first to check length before combining
        prompt_tokens_list = tokenizer.encode(prompt_text, add_special_tokens=False) # Don't add BOS here if template adds it
        completion_tokens_list = tokenizer.encode(completion_text, add_special_tokens=False)

        # Combine tokens respecting chat template logic (simplified: assume prompt needs BOS, completion doesn't)
        # This might need refinement based on the exact template used during generation
        input_ids_list = tokenizer.build_inputs_with_special_tokens(prompt_tokens_list, completion_tokens_list)
        prompt_len_in_combined = len(tokenizer.build_inputs_with_special_tokens(prompt_tokens_list)) # Length of prompt including special tokens added by template

        # Check length constraint
        if len(input_ids_list) > model_max_len:
             logger.warning(f"Sequence length {len(input_ids_list)} exceeds model max length {model_max_len}. Skipping NLL calculation.")
             return None

        # Convert to tensor
        input_ids = torch.tensor([input_ids_list], device=device)

        # Create labels: Ignore prompt tokens (-100), predict completion tokens
        labels = input_ids.clone()
        labels[:, :prompt_len_in_combined] = -100

        # Get model outputs (loss is averaged over sequence)
        outputs = model(input_ids=input_ids, labels=labels)
        avg_nll = outputs.loss.item()

        # Calculate total NLL
        num_predicted_tokens = (labels != -100).sum().item()
        if num_predicted_tokens == 0:
            logger.warning("No completion tokens found after tokenization. Cannot calculate NLL.")
            return None

        total_nll = avg_nll * num_predicted_tokens
        return total_nll

    except Exception as e:
        logger.error(f"Error during NLL calculation: {e}", exc_info=True)
        return None

# --- Accuracy Check Function (Adapted from multi_validate) ---
def check_accuracy(arc_problem: ArcProblem, code: str) -> bool:
     """Checks if the code correctly solves all training pairs for the ARC problem."""
     if not code or not arc_problem or not arc_problem.train_pairs:
         return False # Cannot be correct if no code or no train pairs

     passed_all = True
     for pair in arc_problem.train_pairs:
         try:
             outputs = multi_execute_transformation(
                 codes=[code],
                 inputs=[pair.x], # Input is numpy array
                 timeout=2,
                 function_name="transform"
                 # Add other args like num_workers if needed/possible
             )
             output_grid = outputs[0]

             if isinstance(output_grid, str): # Error/timeout
                 passed_all = False; break
             if not isinstance(output_grid, np.ndarray):
                 passed_all = False; break

             comparison_result, _ = compare_grids(output_grid, pair.y) # pair.y is numpy array
             if comparison_result != GridComparisonResult.EQUAL:
                 passed_all = False; break
         except Exception as e:
             logger.debug(f"Exception during accuracy check for problem {arc_problem.uid}: {e}")
             passed_all = False; break # Treat execution errors as failure
     return passed_all


# --- Main Analysis Function ---
def main():
    parser = argparse.ArgumentParser(description="Analyze ARC solution complexity and accuracy.")
    parser.add_argument("--answer_file", required=True, help="Path to the .jsonl inference file containing generated solutions ('uid', 'prompt', 'responses').")
    parser.add_argument("--nll_model_path", default="codellama/CodeLlama-7b-Python-hf", help="Path or Hugging Face ID of the Code LLM for NLL calculation.")
    parser.add_argument("--output_metrics_file", required=True, help="Path to save the calculated metrics (CSV format).")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device for NLL model ('cuda' or 'cpu').")
    # Add optional argument to limit number of responses per problem for testing
    parser.add_argument("--max_responses_per_problem", type=int, default=None, help="Maximum number of responses to analyze per problem (for debugging/testing).")


    args = parser.parse_args()

    # --- Load NLL Model and Tokenizer ---
    logger.info(f"Loading NLL model: {args.nll_model_path}")
    try:
        # Use trust_remote_code=True if model requires it
        nll_tokenizer = AutoTokenizer.from_pretrained(args.nll_model_path, trust_remote_code=True)
        nll_model = AutoModelForCausalLM.from_pretrained(
            args.nll_model_path,
            torch_dtype=torch.bfloat16,
            device_map=args.device, # Use device_map for simple single-device placement
            trust_remote_code=True
        ).eval()
        nll_model_max_len = getattr(nll_model.config, 'max_position_embeddings', None) # Get max length once
        logger.info(f"NLL model ({args.nll_model_path}) loaded successfully. Max length: {nll_model_max_len}")
    except Exception as e:
        logger.error(f"Failed to load NLL model/tokenizer: {e}", exc_info=True)
        sys.exit(1)

    # --- Load Generated Solutions ---
    if not os.path.exists(args.answer_file):
        raise ValueError(f"Provided --answer_file does not exist: {args.answer_file}")

    logger.info(f"Loading generated solutions from: {args.answer_file}")
    with open(args.answer_file) as f:
        problem_answers = [json.loads(line) for line in f]
    logger.info(f"Loaded {len(problem_answers)} problems.")

    # --- Process Solutions ---
    all_metrics = []
    for p in tqdm(problem_answers, desc="Analyzing solutions"):
        uid = p.get("uid")
        responses = p.get("responses", [])
        original_prompt = p.get("prompt") # Assumed format based on vllm_inference.py

        if not uid or not responses:
            logger.warning(f"Skipping entry due to missing 'uid' or 'responses': {p}")
            continue

        # Retrieve ARC problem data
        arc_problem = None
        try:
            arc_problem = get_arc_problem(uid)
        except ValueError as e:
            logger.warning(f"Could not find ARC problem data for uid {uid}: {e}. Accuracy will be None.")

        # Limit responses if requested
        if args.max_responses_per_problem is not None:
             responses = responses[:args.max_responses_per_problem]

        for response_idx, response_text in enumerate(responses):
            # --- 1. Parse Code ---
            code = parse_code(response_text)
            if not code:
                # Store metrics with code=None and null values for others
                all_metrics.append({
                    "uid": uid, "response_index": response_idx, "is_correct": None,
                    "raw_length": 0, "bit_length": None, "code_parse_error": True
                })
                continue

            # --- 2. Calculate Accuracy ---
            is_correct = None
            if arc_problem:
                 is_correct = 1 if check_accuracy(arc_problem, code) else 0

            # --- 3. Calculate Raw Length ---
            raw_length = len(code)

            # --- 4. Calculate Bit-Length (NLL) ---
            bit_length = None
            if original_prompt:
                total_nll = calculate_nll(original_prompt, code, nll_model, nll_tokenizer, args.device, nll_model_max_len)
                if total_nll is not None:
                    bit_length = total_nll / math.log(2) # Convert to bits
            else:
                 logger.warning(f"Skipping NLL for {uid} response {response_idx} due to missing original prompt in input file.")


            # --- 5. Store Metrics ---
            all_metrics.append({
                "uid": uid,
                "response_index": response_idx,
                "is_correct": is_correct,
                "raw_length": raw_length,
                "bit_length": bit_length,
                "code_parse_error": False
                # Optionally store the code itself: "code": code
            })

    # --- Save Results ---
    logger.info(f"Saving calculated metrics ({len(all_metrics)} total solutions analyzed) to: {args.output_metrics_file}")
    metrics_df = pd.DataFrame(all_metrics)
    try:
        output_dir = os.path.dirname(args.output_metrics_file)
        if output_dir:
             os.makedirs(output_dir, exist_ok=True)
        metrics_df.to_csv(args.output_metrics_file, index=False)
        logger.info("Metrics saved successfully.")
    except Exception as e:
         logger.error(f"Failed to save metrics to CSV: {e}", exc_info=True)


if __name__ == "__main__":
    main()