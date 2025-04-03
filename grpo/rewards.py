# coding=utf-8
# Copyright 2025 The HuggingFace Team. All rights reserved.
# Copyright 2025 BARC Project Authors. All rights reserved. # Added BARC attribution
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Reward functions for GRPO training."""

import asyncio
import json
import math
import re
import logging # Added logging
import numpy as np # Added numpy
from functools import partial, update_wrapper
from typing import Callable, Dict, Optional, List # Adjusted typing

# --- Imports for original open-r1 rewards (math/code/etc) ---
# You might remove these if not using these specific rewards
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify
from .utils import is_e2b_available
from .utils.ioi import SubtaskResult, add_includes, get_piston_client_from_env, score_subtask
# --- End original imports ---

# --- Imports for BARC ARC Accuracy Reward ---
# Assumes these imports work based on BARC structure & your confirmation:
try:
    from utils import parse_code
except ImportError:
    logger.warning("Could not import 'parse_code' from 'utils'. ARC Accuracy reward will fail.")
    parse_code = None

try:
    from execution import multi_execute_transformation
except ImportError:
    logger.warning("Could not import 'multi_execute_transformation' from 'execution'. ARC Accuracy reward will fail.")
    multi_execute_transformation = None

# Importing from root-level file - suggest moving compare_grids to utils later
try:
    from eval_code_samples import compare_grids, GridComparisonResult
except ImportError:
    logger.warning("Could not import 'compare_grids' or 'GridComparisonResult' from 'eval_code_samples'. ARC Accuracy reward will fail.")
    compare_grids = None
    GridComparisonResult = None
# --- End BARC imports ---


if is_e2b_available():
    from dotenv import load_dotenv
    from e2b_code_interpreter import AsyncSandbox
    load_dotenv()
else:
    AsyncSandbox = None

logger = logging.getLogger(__name__)

logger = logging.getLogger(__name__)

# ==========================================================================
# === Helper Functions for Parsing Grids from Prompt Text ===
# ==========================================================================
# Based on BARC README: 0-9 map to Black, Blue, Red, Green, Yellow, Gray, Pink, Orange, Purple, Brown
COLOR_MAP_TEXT_TO_INT = {
    "Black": 0, "Blue": 1, "Red": 2, "Green": 3, "Yellow": 4,
    "Gray": 5, "Pink": 6, "Orange": 7, "Purple": 8, "Brown": 9,
    # Add Grey just in case
    "Grey": 5,
}

def _text_to_grid(grid_text: str) -> Optional[np.ndarray]:
    """Converts a multiline string representation of a grid into a numpy array."""
    try:
        rows = []
        for line in grid_text.strip().split('\n'):
            row = [COLOR_MAP_TEXT_TO_INT[color_name.strip()] for color_name in line.strip().split()]
            rows.append(row)
        return np.array(rows, dtype=int)
    except Exception as e:
        logger.error(f"Failed to parse grid text into numpy array: {e}\nGrid Text:\n{grid_text}")
        return None

def _parse_grids_from_prompt(messages: list[dict]) -> list[dict]:
    """Parses ARC training input/output grids from the text in the user message."""
    train_pairs = []
    user_content = None
    for msg in messages:
        if msg.get("role") == "user":
            user_content = msg.get("content", "")
            break

    if not user_content:
        logger.warning("Could not find user message content to parse grids.")
        return []

    # Simple parsing based on "Example N", "Input:", "Output:" structure
    # Uses regex to find blocks; might need refinement for edge cases.
    example_pattern = re.compile(r"Example \d+.*?Input:(.*?)Output:(.*?)(?=Example \d+|Here is the input grid for the test example:|Write a Python function)", re.DOTALL | re.IGNORECASE)
    input_block_pattern = re.compile(r"Input:(.*)", re.DOTALL | re.IGNORECASE)
    output_block_pattern = re.compile(r"Output:(.*)", re.DOTALL | re.IGNORECASE)

    for match in example_pattern.finditer(user_content):
        input_text_match = input_block_pattern.search(match.group(0))
        output_text_match = output_block_pattern.search(match.group(0))

        if input_text_match and output_text_match:
            input_text = input_text_match.group(1).strip()
            output_text = output_text_match.group(1).strip()

            # Find the actual grid content within these blocks
            # This assumes grid starts after the Input/Output line and ends at the next header or empty lines
            input_grid_text = '\n'.join(line for line in input_text.split('\n') if line.strip() and not line.strip().startswith(('Input:', 'Output:', 'Example')))
            output_grid_text = '\n'.join(line for line in output_text.split('\n') if line.strip() and not line.strip().startswith(('Input:', 'Output:', 'Example')))

            input_grid = _text_to_grid(input_grid_text)
            output_grid = _text_to_grid(output_grid_text)

            if input_grid is not None and output_grid is not None:
                train_pairs.append({"input": input_grid, "output": output_grid})
            else:
                logger.warning(f"Failed to parse one or both grids for an example in prompt.")
                # Decide if partial parsing failure should invalidate the whole prompt
                # For now, we just skip the pair but continue parsing others
        else:
             logger.warning("Could not find Input/Output block within an Example section.")


    if not train_pairs:
         logger.warning("Could not parse any training pairs from the user prompt content.")

    return train_pairs

# ==========================================================================
# === BARC ARC ACCURACY REWARD FUNCTION ===
# ==========================================================================
def accuracy_reward(completions: list[str], messages: list[list[dict]], **kwargs) -> list[float]:
    """
    Calculates reward based on executing the generated Python code (transform function)
    and checking if it correctly transforms all training input grids (parsed from messages)
    to output grids within a timeout.

    Args:
        completions: A list of strings, where each string is the model's generated response
                     containing the Python code.
        messages: A list where each item corresponds to a completion. Each item is the
                  original list of message dictionaries for that example, passed via **kwargs.
        **kwargs: Additional data passed by the GRPOTrainer.

    Returns:
        A list of floats (1.0 for success, 0.0 for failure) for each completion.
    """
    # Check if necessary functions were imported
    if not all([parse_code, multi_execute_transformation, compare_grids, GridComparisonResult]):
         logger.error("ARC Accuracy Reward cannot run because required functions failed to import.")
         return [0.0] * len(completions)

    rewards = []
    batch_size = len(completions)

    # Ensure messages data was passed correctly
    if not messages or len(messages) != batch_size:
         logger.error(f"Mismatch between completions ({batch_size}) and messages ({len(messages) if messages else 0}) length or data missing.")
         return [0.0] * batch_size

    for i in range(batch_size):
        completion = completions[i]
        current_messages = messages[i]

        reward = 0.0 # Default reward is 0.0
        passed_all_train = True

        # --- 1. Parse Training Pairs from Prompt Text ---
        current_train_pairs = []
        try:
             current_train_pairs = _parse_grids_from_prompt(current_messages)
             if not current_train_pairs:
                  logger.warning(f"AccuracyReward: Item {i} - No training pairs parsed from prompt. Assigning 0 reward.")
                  passed_all_train = False # Cannot be accurate if no pairs found/parsed
        except Exception as e:
             logger.error(f"AccuracyReward: Item {i} - Error parsing training pairs from messages: {e}", exc_info=True)
             passed_all_train = False

        # --- 2. Extract Code ---
        code = None
        if passed_all_train: # Only extract code if pairs were parsed
             try:
                  code = parse_code(completion)
                  if not code:
                       logger.debug(f"AccuracyReward: Item {i} - No code found in completion.")
                       passed_all_train = False
             except Exception as e:
                  logger.debug(f"AccuracyReward: Item {i} - Error parsing code: {e}")
                  passed_all_train = False

        # --- 3. Execute Code on Each Parsed Training Pair ---
        if passed_all_train: # Only execute if pairs parsed and code extracted
            for idx, pair_data in enumerate(current_train_pairs):
                try:
                    # Grids should already be numpy arrays from parsing step
                    input_grid = pair_data['input']
                    expected_output = pair_data['output']

                    outputs = multi_execute_transformation(
                        codes=[code],
                        inputs=[input_grid],
                        timeout=2,
                        function_name="transform"
                    )
                    output_grid = outputs[0]

                    # --- 4. Check Execution Result & Compare Grids ---
                    if isinstance(output_grid, str): # Error/timeout
                        logger.debug(f"AccuracyReward: Item {i}, Pair {idx} execution failed/timed out: {output_grid}")
                        passed_all_train = False; break
                    if not isinstance(output_grid, np.ndarray):
                         logger.debug(f"AccuracyReward: Item {i}, Pair {idx} output is not np.ndarray: {type(output_grid)}")
                         passed_all_train = False; break

                    comparison_result, _ = compare_grids(output_grid, expected_output)
                    if comparison_result != GridComparisonResult.EQUAL:
                        logger.debug(f"AccuracyReward: Item {i}, Pair {idx} grid mismatch. Result: {comparison_result}")
                        passed_all_train = False; break

                except Exception as e:
                    logger.error(f"AccuracyReward: Item {i}, Pair {idx} unexpected error during exec/compare: {e}", exc_info=True)
                    passed_all_train = False; break

        # --- 5. Assign Final Reward ---
        if passed_all_train: # This is only True if pairs parsed, code extracted, all pairs executed and matched
            reward = 1.0
            logger.debug(f"AccuracyReward: Item {i} PASSED. Reward: 1.0")
        else:
            # Logged specific failure reasons above
             logger.debug(f"AccuracyReward: Item {i} FAILED. Reward: 0.0")

        rewards.append(reward)

    return rewards

# ==========================================================================
# === OTHER REWARD FUNCTIONS (Copied from open-r1 rewards.py) ===
# ==========================================================================

# Keep format_reward, tag_count_reward, reasoning_steps_reward, len_reward,
# get_cosine_scaled_reward, get_repetition_penalty_reward, ioi_code_reward,
# extract_code, binary_code_reward, code_reward, get_code_format_reward,
# and the E2B helper functions (run_async_from_sync, run_async, run_script)
# IF YOU INTEND TO USE THEM. Otherwise, they can be removed.

# Make sure to keep the get_reward_funcs function and update the registry.

def format_reward(completions, **kwargs):
    """Reward function that checks if the reasoning process is enclosed within <think> and </think> tags, while the final answer is enclosed within <answer> and </answer> tags."""
    # This implementation might need adjustment depending on the exact format expected in BARC's GRPO output
    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
    # Ensure completions are in the expected format (list of strings)
    if completions and isinstance(completions[0], list) and isinstance(completions[0][0], dict):
         completion_contents = [comp[0]["content"] for comp in completions] # Handle potential list nesting
    elif completions and isinstance(completions[0], str):
         completion_contents = completions # Assume already list of strings
    else:
         logger.warning("FormatReward: Unexpected completions format.")
         return [0.0] * len(completions)

    matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]


def tag_count_reward(completions, **kwargs) -> list[float]:
    """Reward function that checks if we produce the desired number of think and answer tags associated with `format_reward()`. (Adapted from Open-R1)"""
    def count_tags(text: str) -> float:
        count = 0.0
        if text.count("<think>\n") == 1: count += 0.25
        if text.count("\n</think>\n") == 1: count += 0.25
        if text.count("\n<answer>\n") == 1: count += 0.25
        if text.count("\n</answer>") == 1: count += 0.25
        return count
        
    if completions and isinstance(completions[0], list) and isinstance(completions[0][0], dict):
         contents = [comp[0]["content"] for comp in completions] # Handle potential list nesting
    elif completions and isinstance(completions[0], str):
         contents = completions # Assume already list of strings
    else:
         logger.warning("TagCountReward: Unexpected completions format.")
         return [0.0] * len(completions)

    return [count_tags(c) for c in contents]

# ...(Keep other reward functions like reasoning_steps_reward, len_reward, etc. if needed)...
# ...(Keep E2B/Piston related functions if using code_reward/ioi_code_reward)...

# ==========================================================================
# === REWARD FUNCTION REGISTRY ===
# ==========================================================================
def get_reward_funcs(script_args) -> list[Callable]:
    """Loads reward functions based on script arguments."""

    REWARD_FUNCS_REGISTRY = {
        "accuracy": accuracy_reward, # Now points to the ARC-specific version
        "format": format_reward, # Keep placeholder/original
        "tag_count": tag_count_reward, # Keep placeholder/original
        "nll_bit_length": None, # Placeholder for your NLL reward
        # Add other keys if needed, remove if not
    }

    active_rewards = {name: func for name, func in REWARD_FUNCS_REGISTRY.items() if func is not None}

    try:
        reward_funcs_to_use = [active_rewards[func_name] for func_name in script_args.reward_funcs]
        logger.info(f"Successfully loaded reward functions: {[getattr(f, '__name__', 'partial_or_unknown') for f in reward_funcs_to_use]}") # Adjusted logging for partials
    except KeyError as e:
        logger.error(f"Reward function '{e}' specified in config not found in registry or not implemented (is None).")
        raise ValueError(f"Invalid or unimplemented reward function name: {e}")
    except AttributeError:
         logger.error("Script arguments object does not have 'reward_funcs' attribute.")
         raise ValueError("Missing 'reward_funcs' in script arguments.")

    return reward_funcs_to_use