#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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

"""
Script to run GRPO (Group Relative Policy Optimization) training for the BARC project,
adapted from the alignment-handbook scripts.
"""

import logging
import random
import sys
# Removed typing import as Any, Dict were not used in ORPO script's main logic directly
import os # Added from open-r1/grpo.py for checkpoint checking

import torch
import transformers
# Added datasets from open-r1/grpo.py for load_dataset (if needed, alternative to get_datasets)
# import datasets
# from datasets import load_dataset
from transformers import AutoModelForCausalLM, set_seed
from transformers.trainer_utils import get_last_checkpoint # Added from open-r1/grpo.py

# Imports from alignment-handbook helpers (kept relevant ones)
from alignment import (
    # DataArguments, # Removed, replaced by GRPOScriptArguments
    H4ArgumentParser,
    ModelArguments,
    # apply_chat_template, # Removed, will use custom GRPO formatting
    # decontaminate_humaneval, # Keep for now, optional
    get_checkpoint, # Keep, but open-r1 uses get_last_checkpoint directly
    get_datasets, # Keep this way of loading data from BARC config
    get_kbit_device_map,
    get_peft_config, # Keep this for consistency if BARC uses it
    get_quantization_config,
    get_tokenizer,
)

# Imports from trl (Updated for GRPO)
# from trl import ORPOConfig, ORPOTrainer, setup_chat_format # Removed ORPO imports
from trl import GRPOTrainer # Added GRPO trainer
# Removed setup_chat_format - handle template manually or verify if needed for Llama3

# Imports from our new BARC.grpo module
from BARC.grpo.configs import GRPOConfig, GRPOScriptArguments # Import GRPO specific configs
from BARC.grpo.rewards import get_reward_funcs # Import reward loading function
# from BARC.grpo.utils.callbacks import get_callbacks # Placeholder for callbacks if needed
# from BARC.grpo.utils.wandb_logging import init_wandb_training # Placeholder if using open-r1 wandb setup

logger = logging.getLogger(__name__)


def main():
    # --- Argument Parsing ---
    # Use GRPOConfig and GRPOScriptArguments instead of ORPOConfig and DataArguments
    parser = H4ArgumentParser((ModelArguments, GRPOScriptArguments, GRPOConfig))
    # Unpack into model_args, script_args (for dataset/reward info), and training_args (for trainer params)
    model_args, script_args, training_args = parser.parse()

    # --- Setup ---
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary (updated variable names):
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.bf16}" # Using bf16 from config
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}") # Log script-specific args
    logger.info(f"Training parameters {training_args}")

    # Check for last checkpoint (using logic from open-r1/grpo.py)
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed for reproducibility
    set_seed(training_args.seed)

    # Initialize W&B potentially (using placeholder, adapt if needed)
    # if "wandb" in training_args.report_to:
    #     init_wandb_training(training_args) # Requires init_wandb_training function

    # --- Load Tokenizer ---
    # Keep using get_tokenizer from alignment helpers for consistency
    tokenizer = get_tokenizer(model_args, data_args=None) # data_args might not be needed if tokenizer config is in model_args/training_args

    # --- Load Reward Functions ---
    # Load rewards based on names provided in script_args (parsed from YAML)
    logger.info(f"Loading reward functions: {script_args.reward_funcs}") # Log reward function names being loaded
    reward_funcs = get_reward_funcs(script_args) # Assumes script_args has 'reward_funcs' attribute

    # === NEXT STEPS ===
    # 1. Load datasets using get_datasets(script_args, ...)
    # (Previous code: imports, arg parsing, setup, tokenizer loading, reward loading...)

    # --- Load Datasets ---
    logger.info(f"Loading datasets specified in: {script_args.dataset_mixer}")
    
    # We know the dataset only has 'messages'. We need to keep it for the reward function.
    # Add 'id' or other metadata columns if they exist and are needed.
    columns_to_keep = ["messages"] 
    # Example: if 'id' column exists: columns_to_keep.append("id")
    
    raw_datasets = get_datasets(
        script_args, 
        splits=script_args.dataset_splits,
        configs=None, 
        columns_to_keep=columns_to_keep, 
    )
    logger.info(
        f"Training on the following splits: {[split + ' : ' + str(dset.num_rows) for split, dset in raw_datasets.items()]}"
    )
    # Get original column names to remove them after mapping
    original_column_names = list(raw_datasets["train"].features)
    if "messages" not in original_column_names:
         # This should not happen based on your inspection, but good to check
         raise ValueError("Dataset loaded by get_datasets does not contain the expected 'messages' column.")

    # --- Process Datasets: Format Prompt and Keep Original Messages ---
    def format_dataset(example):
        # 1. Format the prompt using the 'messages' column
        if "messages" not in example:
            # Should not happen if check above passed, but defensive coding
            raise ValueError("Dataset example missing 'messages' column for prompt formatting.")
        
        formatted_prompt = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=True # Append assistant turn start marker
        )
        
        # 2. Return both the formatted prompt AND the original messages list
        # The original messages will be passed via **kwargs to the reward function
        return {
            "prompt": formatted_prompt, 
            "messages": example["messages"] # Keep the original messages data
        }

    logger.info("Formatting prompt and preserving original 'messages' column...")
    
    # Ensure GRPOScriptArguments defines preprocessing_num_workers
    num_proc = getattr(script_args, "preprocessing_num_workers", None) 
    
    raw_datasets = raw_datasets.map(
        format_dataset,
        num_proc=num_proc, 
        remove_columns=original_column_names, # Remove original 'messages' column
        desc="Formatting prompt and preserving original messages",
    )
    
    # The dataset now has 'prompt' and 'messages' columns (plus 'id' etc. if kept)
    
    # Log a few random samples to verify
    for index in random.sample(range(len(raw_datasets["train"])), min(3, len(raw_datasets["train"]))): 
        logger.debug(f"Prompt sample {index}:\n\n{raw_datasets['train'][index]['prompt']}")
        # logger.debug(f"Messages sample {index}:\n\n{raw_datasets['train'][index]['messages']}") # Optional: log preserved messages


    # --- Load Model ---
    logger.info(f"Loading model: {model_args.model_name_or_path}")
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    
    # Note: Quantization config and device map are omitted based on user input (no quantization, single GPU)
    # If running multi-GPU with accelerate/deepspeed, device mapping is typically handled by the launcher.
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        # device_map=None, # Explicitly setting to None or omitting for single GPU / Accelerate handling
        # quantization_config=None, # Omitted as quantization is not used
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        **model_kwargs,
    )
    logger.info("Model loaded successfully.")

    # --- Instantiate GRPOTrainer ---
    logger.info("*** Initializing GRPOTrainer ***")

    # Get PEFT config if required (standard practice in these scripts)
    peft_config = get_peft_config(model_args)

    # Placeholder for callbacks - adapt or remove if not using get_callbacks from open-r1/alignment
    # Ensure get_callbacks is defined and imported, e.g., from BARC.grpo.utils.callbacks
    # callbacks = get_callbacks(training_args, model_args)
    callbacks = [] # Use empty list if no callbacks for now

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_funcs, # The list loaded earlier via get_reward_funcs
        args=training_args, # The parsed GRPOConfig object
        train_dataset=raw_datasets["train"], # Use appropriate split name
        eval_dataset=raw_datasets["test"] if training_args.do_eval else None, # Use appropriate split name
        tokenizer=tokenizer,
        peft_config=peft_config,
        max_prompt_length=training_args.max_prompt_length, # From GRPOConfig/YAML
        max_completion_length=training_args.max_completion_length, # From GRPOConfig/YAML
        num_generations=training_args.num_generations, # From GRPOConfig/YAML
        callbacks=callbacks, # Pass the callbacks list
    )
    logger.info("GRPOTrainer initialized successfully.")
    
    # --- Training Loop ---
    logger.info("*** Train ***")
    checkpoint = None
    # Logic for resuming from checkpoint (kept from run_orpo.py/open-r1/grpo.py)
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
        
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    # Add number of samples metric using the correct dataset split name
    metrics["train_samples"] = len(raw_datasets[script_args.dataset_train_split]) 
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("*** Training complete ***")

    # --- Save Model and Create Model Card ---
    logger.info("*** Save model ***")
    # Handle FSDP saving if necessary (copied from run_orpo.py)
    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
        
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process (Logic from run_orpo.py, adjust kwargs)
    # Determine dataset name(s) based on script_args (which parsed GRPOScriptArguments)
    # Assuming GRPOScriptArguments has dataset_mixer or dataset_name
    dataset_name_for_card = ""
    if hasattr(script_args, "dataset_mixer") and script_args.dataset_mixer:
         dataset_name_for_card = list(script_args.dataset_mixer.keys())
    elif hasattr(script_args, "dataset_name") and script_args.dataset_name:
        dataset_name_for_card = script_args.dataset_name
        
    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "dataset": dataset_name_for_card, # Use determined dataset name(s)
        "dataset_tags": dataset_name_for_card, # Use determined dataset name(s)
        "tags": ["alignment-handbook", "grpo", "barc"], # Add relevant tags
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        # Note: GRPOTrainer might handle this differently or it might conflict with vLLM backend.
        # Check if needed/compatible. Commenting out for now.
        # trainer.model.config.use_cache = True 
        # trainer.model.config.save_pretrained(training_args.output_dir)
        
        # Save tokenizer
        tokenizer.save_pretrained(training_args.output_dir)

    # --- Evaluate ---
    # Use the correct eval dataset split name from script_args
    eval_split_name = script_args.dataset_test_split
    if training_args.do_eval and eval_split_name in raw_datasets:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(raw_datasets[eval_split_name])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # --- Push to Hub ---
    if training_args.push_to_hub: # Checks the flag in your config
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs) # Pass model card kwargs

    logger.info("*** GRPO Script complete! ***")
    


if __name__ == "__main__":
    main()