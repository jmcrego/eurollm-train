import os
import torch
import logging
import argparse
from pathlib import Path
from sacrebleu import corpus_bleu
from transformers import TrainerCallback
from transformers import AutoTokenizer, AutoModelForCausalLM
from EuroLLMDataset import create_eval_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to continue Pre-Training EuroLLM models.")
    parser.add_argument("--model_path", type=str, default="/lustre/fsmisc/dataset/HuggingFace_Models/utter-project/EuroLLM-1.7B", help="Path to the EuroLLM model.")
    parser.add_argument("--eval_config", type=str, required=True, help="Eval json config file.")
    parser.add_argument("--target_language_sacrebleu", type=str, default=None, help="target language for sacrebleu string tokenizer.")
    parser.add_argument("--per_device_batch_size", type=int, default=16, help="Per device batch size.")
    parser.add_argument("--mask_id", type=int, default=-100, help="Id used to mask the prompt when learning (compute loss over target only).")
    parser.add_argument("--max_length", type=int, default=512, help="Max sequence length.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for randomness in training dataset/s.")
    parser.add_argument("--bf16", action='store_true', help="Use torch_dtype=torch.bfloat16.")
    parser.add_argument("--fp16", action='store_true', help="Use torch_dtype=torch.float16.")
    args = parser.parse_args()

    # === LOAD TOKENIZER ===
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token
    logger.info(f"Loaded tokenizer {args.model_path}")
    logger.info(f"Tokenizer BOS token id: {tokenizer.bos_token_id}: {tokenizer.bos_token}")
    logger.info(f"Tokenizer EOS token id: {tokenizer.eos_token_id}: {tokenizer.eos_token}")
    logger.info(f"Tokenizer PAD token id: {tokenizer.pad_token_id}: {tokenizer.pad_token}")
    logger.info(f"Tokenizer padding_side: {tokenizer.padding_side}")
    #set_seed

    if args.bf16:
        if torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        else:
            logger.warning("BF16 requested but not supported on this hardware. Falling back to float32.")
            dtype = torch.float32
    elif args.fp16:
        dtype = torch.float16
    else:
        dtype = torch.float32
    logger.info(f"dtype is {dtype}")

    # === LOAD MODEL ===
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=dtype, device_map="auto")
    model.config.pad_token_id = tokenizer.pad_token_id
    logger.info(f"Loaded model from pretrained {args.model_path}")
    logger.info(f"Model dtype={next(model.parameters()).dtype}")
    logger.info(f"Model config BOS token id: {model.config.bos_token_id}")
    logger.info(f"Model config PAD token id: {model.config.pad_token_id}")

    # === LOAD DATASET ===
    eval_dataset = create_eval_dataset(args.eval_config, tokenizer, batch_size=args.per_device_batch_size, mask_id=args.mask_id)

    # === GREEDY GENERATIION ===
    bleu = eval_greedy(model, tokenizer, eval_dataset, args.max_length, args.per_device_batch_size, "./kkout", target_language=args.target_language_sacrebleu)
    logger.info(f"SacreBLEU : {bleu}")
