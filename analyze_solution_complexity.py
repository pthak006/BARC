#!/usr/bin/env python
# coding=utf-8

"""
Analyzes generated ARC solutions from a JSONL file.
Calculates accuracy (based on training pairs from a pre-computed exec_results file),
raw length (characters), and bit-length (NLL using a specified Code LLM).
Saves the results to a CSV file incrementally.

Based on the original eval_code_samples.py from the BARC repository.
Changes:
- 'is_correct' is now inferred from an exec_results_v4.jsonl file.
- 'code_parse_error' column and its calculation have been removed.
- Results are saved incrementally after each problem is processed.
"""

import os
import sys
import json
import argparse
import math
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import logging
from typing import Optional, Dict, List, Any
import pathlib # Added

# --- Imports from BARC ---
# Assume these are importable from the BARC project structure
try:
    from utils import parse_code
except ImportError:
    print("ERROR: Could not import 'parse_code' from 'utils'. Ensure BARC utils are in PYTHONPATH.")
    sys.exit(1)
# Removed imports related to multi_validate as it's no longer used here:
# from execution import multi_execute_transformation
# from arc import train_problems, validation_problems, ArcProblem, ArcIOPair, get_arc_problem
# Removed GridComparisonResult, compare_grids as they were part of multi_validate

# --- End BARC Imports/Setup ---


# --- Logging Setup ---
LOG_FILENAME = 'eval_code_samples_analysis_revised.log' # Changed log filename
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
METRICS_LOG_FILENAME = 'metrics_details_revised.log'
metrics_logger = logging.getLogger('metrics_logger')
metrics_logger.setLevel(logging.INFO)
metrics_logger.propagate = False  # Prevent propagation to root logger
metrics_file_handler = logging.FileHandler(METRICS_LOG_FILENAME)
metrics_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s', datefmt="%Y-%m-%d %H:%M:%S"))
metrics_logger.addHandler(metrics_file_handler)
logger.info(f"Metrics details will be logged to: {METRICS_LOG_FILENAME}")
# --- End Logging Setup ---


# --- NLL Calculation Function (Kept from original) ---
@torch.no_grad()
def calculate_nll(prompt_text: str, completion_text: str, model, tokenizer, device='cuda', model_max_len=None) -> Optional[float]:
    """
    Calculates the total negative log-likelihood of the completion_text given the prompt_text.
    Uses standard tokenizer call and handles prompt/completion separation for labeling.
    """
    if model_max_len is None:
        model_max_len = getattr(model.config, 'max_position_embeddings', 2048)

    try:
        prompt_encoding = tokenizer(prompt_text, add_special_tokens=False, return_tensors="pt", truncation=False)
        completion_encoding = tokenizer(completion_text, add_special_tokens=False, return_tensors="pt", truncation=False)

        prompt_ids = prompt_encoding['input_ids']
        completion_ids = completion_encoding['input_ids']

        input_ids = torch.cat([prompt_ids, completion_ids], dim=-1)

        if input_ids.shape[-1] > model_max_len:
            logger.warning(f"Sequence length {input_ids.shape[-1]} exceeds NLL model max length {model_max_len}. Truncating.")
            input_ids = input_ids[:, :model_max_len]
            completion_len = max(0, model_max_len - prompt_ids.shape[-1])
        else:
            completion_len = completion_ids.shape[-1]

        if prompt_ids.shape[-1] >= model_max_len:
             logger.warning(f"Prompt length {prompt_ids.shape[-1]} exceeds or equals model max length {model_max_len}. Cannot calculate completion NLL.")
             return None

        labels = input_ids.clone()
        labels[:, :prompt_ids.shape[-1]] = -100

        if labels.shape[-1] > model_max_len:
             labels = labels[:, :model_max_len]

        num_predicted_tokens = (labels != -100).sum().item()
        if num_predicted_tokens == 0:
            logger.warning(f"No completion tokens left after prompt/truncation ({completion_len} tokens). Cannot calculate NLL.")
            return None

        input_ids = input_ids.to(device)
        labels = labels.to(device)

        outputs = model(input_ids=input_ids, labels=labels)
        avg_nll = outputs.loss.item()

        if not math.isfinite(avg_nll):
             logger.warning(f"NLL calculation resulted in non-finite value ({avg_nll}). Skipping.")
             return None

        total_nll = avg_nll * num_predicted_tokens
        return total_nll

    except Exception as e:
        logger.error(f"Error during NLL calculation: {e}", exc_info=True)
        return None


# --- Main Analysis Function ---
def main():
    parser = argparse.ArgumentParser(description="Analyze ARC solution complexity and accuracy.")
    parser.add_argument("--answer_file", required=True, help="Path to the .jsonl inference file containing generated solutions ('uid', 'prompt', 'responses').")
    parser.add_argument("--exec_results_file", required=True, help="Path to the corresponding _exec_results_v4.jsonl file produced by eval_code_samples.py.")
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
            torch_dtype=torch.bfloat16, # Consider making this configurable or checking model capabilities
            device_map=args.device, # Changed from device_map="auto"
            trust_remote_code=True
        ).eval()
        nll_model_max_len = getattr(nll_model.config, 'max_position_embeddings', None)
        logger.info(f"NLL model ({args.nll_model_path}) loaded successfully. Max length: {nll_model_max_len}")
    except Exception as e:
        logger.error(f"Failed to load NLL model/tokenizer: {e}", exc_info=True)
        sys.exit(1)

    # --- Load Generated Solutions ---
    if not os.path.exists(args.answer_file):
        logger.error(f"Provided --answer_file does not exist: {args.answer_file}")
        sys.exit(1)

    logger.info(f"Loading generated solutions from: {args.answer_file}")
    with open(args.answer_file) as f:
        problem_answers = [json.loads(line) for line in f]
    logger.info(f"Loaded {len(problem_answers)} problems from answers file.")

    # --- Load Execution Results for Correctness ---
    if not os.path.exists(args.exec_results_file):
        logger.error(f"Provided --exec_results_file does not exist: {args.exec_results_file}")
        sys.exit(1)

    logger.info(f"Loading execution results from: {args.exec_results_file}")
    exec_results_map: Dict[str, List[bool]] = {}
    try:
        with open(args.exec_results_file) as f:
            for line in f:
                data = json.loads(line)
                uid = data.get("uid")
                train_verdicts = data.get("train_verdicts")
                if uid and isinstance(train_verdicts, list):
                    exec_results_map[uid] = train_verdicts
                else:
                    logger.warning(f"Skipping line in exec_results_file due to missing uid or invalid train_verdicts: {line.strip()}")
        logger.info(f"Loaded execution results for {len(exec_results_map)} UIDs.")
    except Exception as e:
        logger.error(f"Failed to load or parse exec_results_file: {e}", exc_info=True)
        sys.exit(1)

    # --- Initialize Output CSV File ---
    output_dir_path = pathlib.Path(args.output_metrics_file).parent
    if output_dir_path:
        output_dir_path.mkdir(parents=True, exist_ok=True)
    
    # Create CSV with headers
    columns = ["uid", "response_index", "is_correct", "raw_length", "bit_length"]
    pd.DataFrame(columns=columns).to_csv(args.output_metrics_file, index=False)
    logger.info(f"Initialized output CSV file: {args.output_metrics_file}")

    # --- Process Solutions ---
    solved_problem_count = 0
    total_problems = len(problem_answers)

    for problem_idx, p_answer in enumerate(tqdm(problem_answers, desc="Analyzing code solutions")):
        uid = p_answer.get("uid")
        responses = p_answer.get("responses", [])
        original_prompt = p_answer.get("prompt")
        problem_metrics = []  # Temporary list for current problem's metrics

        current_problem_processed_index = problem_idx + 1

        if not uid:
            logger.warning(f"Skipping entry {problem_idx} due to missing 'uid'.")
            logger.info(f"Problems Solved So Far: {solved_problem_count}/{current_problem_processed_index}")
            continue

        if args.max_responses_per_problem is not None:
             responses = responses[:args.max_responses_per_problem]

        problem_exec_verdicts = exec_results_map.get(uid)
        if problem_exec_verdicts is None:
            logger.warning(f"No execution results found for UID {uid} in exec_results_file. 'is_correct' will be None for all its responses.")

        problem_solved_by_any_response = False

        for response_idx, response_text in enumerate(responses):
            # Parse code first
            parsed_code_info = parse_code(response_text)
            code = parsed_code_info[0] if parsed_code_info else ""

            # Determine Accuracy from exec_results_map
            is_correct = None
            if problem_exec_verdicts:
                if response_idx < len(problem_exec_verdicts):
                    is_correct = 1 if problem_exec_verdicts[response_idx] else 0
                    if is_correct == 1:
                        problem_solved_by_any_response = True
                else:
                    logger.warning(f"Response index {response_idx} out of bounds for UID {uid} in exec_results_file (found {len(problem_exec_verdicts)} verdicts). Setting is_correct to None.")

            # Calculate Raw Length
            raw_length = len(code)

            # Calculate Bit-Length (NLL)
            bit_length = None
            if code and original_prompt:
                total_nll = calculate_nll(original_prompt, code, nll_model, nll_tokenizer, args.device, nll_model_max_len)
                if total_nll is not None:
                    bit_length = total_nll / math.log(2)
            elif not original_prompt and code:
                 logger.warning(f"Skipping NLL for {uid} response {response_idx} due to missing original prompt.")
            elif not code:
                 logger.debug(f"Skipping NLL for {uid} response {response_idx} as no code was parsed.")

            # Store Metrics
            metric_entry = {
                "uid": uid,
                "response_index": response_idx,
                "is_correct": is_correct,
                "raw_length": raw_length,
                "bit_length": bit_length,
            }
            problem_metrics.append(metric_entry)

            # Log metrics to separate log file
            metrics_logger.info(f"Problem: {uid}, Response: {response_idx}, Correct: {is_correct}, "
                               f"Length: {raw_length}, Bit-Length: {bit_length}")

        # Save metrics for this problem
        if problem_metrics:
            problem_df = pd.DataFrame(problem_metrics)
            problem_df.to_csv(args.output_metrics_file, mode='a', header=False, index=False)
            logger.debug(f"Saved metrics for problem {uid}")
        
        # Clear problem metrics from memory
        problem_metrics = []

        if problem_solved_by_any_response:
            solved_problem_count += 1
        logger.info(f"Problems Solved So Far: {solved_problem_count}/{current_problem_processed_index}")

    # --- Final Summary ---
    logger.info(f"Finished processing. Total problems solved: {solved_problem_count}/{total_problems}")
    logger.info("Metrics have been saved incrementally to the CSV file.")

if __name__ == "__main__":
    main()