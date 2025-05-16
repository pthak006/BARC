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
import traceback
import sys
import json
from enum import Enum
import argparse
import math # Added
import numpy as np
import pandas as pd # Added
import torch # Added
from transformers import AutoModelForCausalLM, AutoTokenizer # Added
from tqdm import tqdm
import logging # Added
from typing import Optional, Dict, List # Added

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
    # Using functions/classes directly from original script structure
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
    from arc import train_problems, validation_problems, ArcProblem, ArcIOPair # Import ArcIOPair if needed
    # Combine problems
    all_arc_problems_dict = {p.uid: p for p in (train_problems + validation_problems)}
    def get_arc_problem(uid):
        problem = all_arc_problems_dict.get(uid)
        if problem is None:
             raise ValueError(f"Problem {uid} not found in preloaded train/validation sets.")
        return problem

    # multi_validate function kept from original eval_code_samples.py
    def multi_validate(arc_problem, codes):
        """
        For each snippet in codes, and for each input (both train & test),
        produce:
            results[code_idx][pair_idx] = (boolEqual, ratio, output_grid_as_ndarray or error).
        """
        # Ensure pairs are ArcIOPair objects if expected by multi_execute_transformation
        if not isinstance(arc_problem.train_pairs, list) or not isinstance(arc_problem.test_pairs, list):
             raise TypeError("arc_problem.train_pairs and .test_pairs must be lists")
             
        pairs = arc_problem.train_pairs + arc_problem.test_pairs
        results = [[] for _ in range(len(codes))]

        for pair_idx, pair in enumerate(pairs):
            input_grid = pair.x
            # Removed TRANSPOSE logic for simplicity, assume False
            try:
                output_grids = multi_execute_transformation(
                    codes, [input_grid]*len(codes),
                    random_seeds=[0]*len(codes),
                    timeout=2,
                    function_name="transform",
                    num_workers=64 # Keep or adjust as needed
                )
            except KeyboardInterrupt:
                sys.exit()
            except Exception as e:
                logger.error(f"multi_execute error for uid {arc_problem.uid}, pair {pair_idx}: {e}", exc_info=False)
                output_grids = ["error"]*len(codes)

            expected_output = pair.y
            for code_idx, out_grid in enumerate(output_grids):
                if isinstance(out_grid, str):
                    results[code_idx].append((False, 0.0, out_grid))
                    continue

                comparison_result, ratio = compare_grids(out_grid, expected_output)
                is_equal = (comparison_result == GridComparisonResult.EQUAL)
                results[code_idx].append((is_equal, ratio, out_grid))
        return results

except ImportError as e:
    print(f"ERROR: Could not import required components: {e}. Ensure BARC environment is set up.")
    sys.exit(1)
# --- End BARC Imports/Setup ---


# --- Logging Setup ---
LOG_FILENAME = 'eval_code_samples_analysis.log' # Changed log filename
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILENAME), # Log to a file
        logging.StreamHandler(sys.stdout)  # Also log to the console
    ]
)
logger = logging.getLogger(__name__)
logger.info(f"Logging to console and file: {LOG_FILENAME}")

# Setup a separate logger for metrics that only logs to a file
METRICS_LOG_FILENAME = 'metrics_details.log'
metrics_logger = logging.getLogger('metrics_logger')
metrics_logger.setLevel(logging.INFO)
metrics_logger.propagate = False  # Prevent propagation to root logger
metrics_file_handler = logging.FileHandler(METRICS_LOG_FILENAME)
metrics_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s', datefmt="%Y-%m-%d %H:%M:%S"))
metrics_logger.addHandler(metrics_file_handler)
logger.info(f"Metrics details will be logged to: {METRICS_LOG_FILENAME}")
# --- End Logging Setup ---


# --- NLL Calculation Function (Reverted to previous version based on git diff) ---
@torch.no_grad() # Ensure no gradients are calculated during NLL computation
def calculate_nll(prompt_text: str, completion_text: str, model, tokenizer, device='cuda', model_max_len=None) -> Optional[float]:
    """
    Calculates the total negative log-likelihood of the completion_text given the prompt_text.
    Uses standard tokenizer call and handles prompt/completion separation for labeling.
    """
    # Use the model's configured max length if not provided
    if model_max_len is None:
        model_max_len = getattr(model.config, 'max_position_embeddings', 2048) # Default to 2048 if not found

    try:
        # 1. Tokenize prompt and completion separately to find prompt length
        # Important: Use add_special_tokens=False initially if the model's template logic
        # is handled by build_inputs_with_special_tokens later or if we handle manually.
        # Let's assume the prompt already has necessary BOS etc. from apply_chat_template
        # and completion should be treated as raw text.
        prompt_encoding = tokenizer(prompt_text, add_special_tokens=False, return_tensors="pt", truncation=False)
        completion_encoding = tokenizer(completion_text, add_special_tokens=False, return_tensors="pt", truncation=False)

        prompt_ids = prompt_encoding['input_ids']
        completion_ids = completion_encoding['input_ids']

        # 2. Combine into a single sequence
        # Note: Depending on the model/tokenizer, special tokens (like BOS/EOS) might need
        # careful handling. This assumes prompt already contains start tokens and we append completion.
        input_ids = torch.cat([prompt_ids, completion_ids], dim=-1)

        # 3. Check total length against model max length
        if input_ids.shape[-1] > model_max_len:
            logger.warning(f"Sequence length {input_ids.shape[-1]} exceeds NLL model max length {model_max_len}. Truncating.")
            # Truncate from the right (affects NLL calculation, but necessary)
            input_ids = input_ids[:, :model_max_len]
            # Adjust completion tokens length if truncation happened
            completion_len = max(0, model_max_len - prompt_ids.shape[-1])
        else:
            completion_len = completion_ids.shape[-1]

        # If prompt itself was longer than max length
        if prompt_ids.shape[-1] >= model_max_len:
             logger.warning(f"Prompt length {prompt_ids.shape[-1]} exceeds or equals model max length {model_max_len}. Cannot calculate completion NLL.")
             return None

        # 4. Create labels: ignore prompt tokens (-100)
        labels = input_ids.clone()
        labels[:, :prompt_ids.shape[-1]] = -100 # Ignore all prompt tokens

        # Ensure labels are correctly truncated if input_ids were truncated
        if labels.shape[-1] > model_max_len:
             labels = labels[:, :model_max_len]

        # Ensure there are actual completion tokens left to predict
        num_predicted_tokens = (labels != -100).sum().item()
        if num_predicted_tokens == 0:
            logger.warning(f"No completion tokens left after prompt/truncation ({completion_len} tokens). Cannot calculate NLL.")
            return None

        # Move tensors to the correct device
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        # 5. Get model outputs (loss is averaged over sequence where labels != -100)
        outputs = model(input_ids=input_ids, labels=labels)
        avg_nll = outputs.loss.item() # Loss is the average CrossEntropyLoss

        # Check for NaN/Inf loss
        if not math.isfinite(avg_nll):
             logger.warning(f"NLL calculation resulted in non-finite value ({avg_nll}). Skipping.")
             return None

        # 6. Calculate total NLL
        # Use the actual number of predicted tokens for calculating total NLL
        total_nll = avg_nll * num_predicted_tokens
        return total_nll

    except Exception as e:
        logger.error(f"Error during NLL calculation: {e}", exc_info=True)
        return None


# --- Main Analysis Function ---
def main():
    # Modified Argument Parser
    parser = argparse.ArgumentParser(description="Analyze ARC solution complexity and accuracy by modifying eval_code_samples.")
    parser.add_argument("--answer_file", required=True, help="Path to the .jsonl inference file containing generated solutions ('uid', 'prompt', 'responses').")
    parser.add_argument("--nll_model_path", default="codellama/CodeLlama-7b-Python-hf", help="Path or Hugging Face ID of the Code LLM for NLL calculation.")
    parser.add_argument("--output_metrics_file", required=True, help="Path to save the calculated metrics (CSV format).")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device for NLL model ('cuda' or 'cpu').")
    parser.add_argument("--max_responses_per_problem", type=int, default=None, help="Maximum number of responses to analyze per problem (for debugging/testing).")
    args = parser.parse_args()

    # --- Load NLL Model and Tokenizer ---
    logger.info(f"Loading NLL model: {args.nll_model_path}")
    try:
        nll_tokenizer = AutoTokenizer.from_pretrained(args.nll_model_path, trust_remote_code=True)
        nll_model = AutoModelForCausalLM.from_pretrained(
            args.nll_model_path,
            torch_dtype=torch.bfloat16,
            device_map=args.device,
            trust_remote_code=True
        ).eval()
        nll_model_max_len = getattr(nll_model.config, 'max_position_embeddings', None)
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
    solved_problem_count = 0
    total_problems = len(problem_answers)

    for problem_idx, p in enumerate(tqdm(problem_answers, desc="Analyzing code solutions")): # Changed description
        uid = p.get("uid")
        responses = p.get("responses", [])
        original_prompt = p.get("prompt") # Assumed format

        current_problem_processed_index = problem_idx + 1

        if not uid or not responses:
            logger.warning(f"Skipping entry {problem_idx} due to missing 'uid' or 'responses'.")
            logger.info(f"Problems Solved So Far: {solved_problem_count}/{current_problem_processed_index}") # Log progress
            continue

        # Retrieve ARC problem data
        arc_problem = None
        try:
            arc_problem = get_arc_problem(uid)
            n_train = len(arc_problem.train_pairs)
        except ValueError as e:
            logger.warning(f"Could not find ARC problem data for uid {uid}: {e}. Accuracy will be None.")
            n_train = 0 # Need a default value if arc_problem is None

        # Limit responses if requested
        if args.max_responses_per_problem is not None:
             responses = responses[:args.max_responses_per_problem]

        # Parse all codes first (original logic from eval_code_samples)
        codes = []
        response_texts_parsed = [] # Store corresponding response text for NLL calc later
        for response_text in responses:
            parsed = parse_code(response_text)
            codes.append(parsed[0] if parsed else "") # Append code string or empty string
            response_texts_parsed.append(response_text) # Keep original response text

        # Run multi_validate once for all codes for this problem (original logic)
        results = []
        if arc_problem and any(c for c in codes): # Only run validation if problem data exists and at least one code was parsed
             try:
                 results = multi_validate(arc_problem, codes)
             except Exception as e:
                  logger.error(f"Error during multi_validate for {uid}: {e}", exc_info=False)
                  # results will remain empty or partially filled, subsequent loop handles this

        problem_solved_by_any_response = False

        # Process results for each code
        for code_idx, code in enumerate(codes):
            # Determine Accuracy from multi_validate results
            is_correct = None
            code_parse_error_flag = (code == "")

            if not code_parse_error_flag and arc_problem and code_idx < len(results) and results[code_idx]:
                # results[code_idx] contains list of results per pair
                result_for_code = results[code_idx]
                # Check if passed all *training* pairs
                is_correct = 1 if all(item[0] for item in result_for_code[:n_train]) else 0
                if is_correct == 1:
                    problem_solved_by_any_response = True
            elif not arc_problem:
                 is_correct = None # Cannot determine accuracy
                 
            # Calculate Raw Length (Fixed)
            raw_length = len(code) if not code_parse_error_flag else 0

            # Calculate Bit-Length (NLL)
            bit_length = None
            if not code_parse_error_flag and original_prompt: # Only if code was parsed and prompt exists
                total_nll = calculate_nll(original_prompt, code, nll_model, nll_tokenizer, args.device, nll_model_max_len)
                if total_nll is not None:
                    bit_length = total_nll / math.log(2)
            elif not original_prompt:
                 logger.warning(f"Skipping NLL for {uid} response {code_idx} due to missing original prompt.")

            # Store Metrics
            metric_entry = {
                "uid": uid,
                "response_index": code_idx,
                "is_correct": is_correct,
                "raw_length": raw_length,
                "bit_length": bit_length,
                "code_parse_error": code_parse_error_flag
            }
            all_metrics.append(metric_entry)
            
            # Log metrics to separate log file
            metrics_logger.info(f"Problem: {uid}, Response: {code_idx}, Correct: {is_correct}, " 
                               f"Length: {raw_length}, Bit-Length: {bit_length}, Parse Error: {code_parse_error_flag}")
        # End inner loop (code_idx)

        # Update Solved Problem Count & Log Progress
        if problem_solved_by_any_response:
            solved_problem_count += 1
        logger.info(f"Problems Solved So Far: {solved_problem_count}/{current_problem_processed_index}")
        # End outer loop (problem_idx)

    # --- Final Summary ---
    logger.info(f"Finished processing. Total problems solved: {solved_problem_count}/{total_problems}")

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

    # Remove original JSONL saving logic from eval_code_samples.py
    # logger.info(f"[eval_code_samples] Saving to {saving_file}") # Original output path logic removed
    # ... (original saving loop removed) ...


if __name__ == "__main__":
    main()