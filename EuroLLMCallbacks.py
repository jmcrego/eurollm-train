import os
import torch
import logging
from pathlib import Path
from sacrebleu import corpus_bleu
from transformers import TrainerCallback
#from torch.utils.data import DataLoader
        
logger = logging.getLogger("training")

class LogStepCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        logs = logs or {}
        step = state.global_step
        if 'eval_loss' in logs:
            logger.info(f"[Eval  @ step {step}]: {logs}")
        elif state.is_local_process_zero:
            logger.info(f"[Train @ step {step}]: {logs}")

class SaveTokenizerCallback(TrainerCallback):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def on_save(self, args, state, control, **kwargs):
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        self.tokenizer.save_pretrained(checkpoint_dir) # Save the tokenizer to the checkpoint directory
            
class CustomBLEUCallback(TrainerCallback):
    def __init__(self, tokenizer, eval_dataset, max_length, batch_size, target_language=None):
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.max_length = max_length
        self.batch_size = batch_size
        self.target_language = target_language

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        output_file = f"{args.output_dir}/eval.{state.global_step}"
        bleu = eval_greedy(model, self.tokenizer, self.eval_dataset, max_length=self.max_length, batch_size=self.batch_size, output_file=output_file, target_language=self.target_language)
        logger.info(f"[Eval @ step {state.global_step}] SacreBLEU: {bleu:.2f}")
        # Log the BLEU score into Trainer state
        state.log_history.append({
            "eval_sacrebleu": bleu,
            "step": state.global_step,
        })
        return control

def eval_greedy(model, tokenizer, eval_dataset, max_length, batch_size, output_file, target_language=None):
    model.eval()
    device = next(model.parameters()).device
    refs = []
    hyps = []

    with torch.no_grad():
        n = 0
        for i, batch in enumerate(eval_dataset):
            batch = {k: v.to(device) for k, v in batch.items()}
            input_ids = batch["prompt_ids"]
            attention_mask = batch["prompt_mask"]
            labels = batch["references"]
            
            prompt_len = len(input_ids[0])
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            attention_mask = attention_mask.to(device)
            generated_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=max_length, pad_token_id=tokenizer.pad_token_id, do_sample=False)
            for j in range(input_ids.size(0)):
                n += 1
                # === strings ===
                input_str = tokenizer.decode(input_ids[j], skip_special_tokens=True).replace("\n", "⏎")
                prediction_str = tokenizer.decode(generated_ids[j][prompt_len:], skip_special_tokens=True).replace("\n", "⏎") #remove prompt
                reference_str = tokenizer.decode(labels[j], skip_special_tokens=True).replace("\n", "⏎")
                refs.append(reference_str)
                hyps.append(prediction_str)

                logger.info(f"[Sample {n}]")
                #logger.info(f"Prompt [{len(input_ids[j])}]: {input_ids[j]}")
                #logger.info(f"Mask [{len(attention_mask[j])}]: {attention_mask[j]}")
                #logger.info(f"Ref    [{len(labels[j])}]: {labels[j]}")
                #logger.info(f"Hyp    [{len(generated_ids[j])}]: {generated_ids[j]}")
                #logger.info(f"Prompt : {input_str}")
                logger.info(f"Ref    : {reference_str}")
                logger.info(f"Hyp    : {prediction_str}")

    # BLEU computation
    bleu = 0.00
    if len(hyps) > 0:
        bleu = corpus_bleu(hyps, [refs], tokenize=get_bleu_tokenizer(target_language)).score

    path = Path(f"{output_file}.{bleu:.2f}.out")    
    if path.parent != Path("."): # Create directory only if it's not the current one
        path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f: # Write hypotheses
        for hyp in hyps:
            f.write(hyp + "\n")

    return bleu

def get_bleu_tokenizer(target_language):
    if target_language is None:
        return "none"
    elif target_language.lower() in {"ja", "japanese"}:
        return "ja-mecab" #requires: pip install "sacrebleu[ja]"
    elif target_language.lower() in {"zh", "chinese"}:
        return "zh" #requires pip install "sacrebleu[zh]"
    else:
        return "none"  # fallback if texts are pre-tokenized


if __name__ == "__main__":
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from EuroLLMDataset import create_eval_dataset
    import argparse

    parser = argparse.ArgumentParser(description="Script to continue Pre-Training EuroLLM models.")
    parser.add_argument("--model_path", type=str, default="/lustre/fsmisc/dataset/HuggingFace_Models/utter-project/EuroLLM-1.7B", help="Path to the EuroLLM model.")
    parser.add_argument("--eval_config", type=str, default="data/eval/eval.json", help="Eval json config file.")
    parser.add_argument("--per_device_batch_size", type=int, default=16, help="Per device batch size.")
    parser.add_argument("--mask_id", type=int, default=-100, help="Id used to mask the prompt when learning (compute loss over target only).")
    parser.add_argument("--target_language_sacrebleu", type=str, default=None, help="target language for sacrebleu string tokenizer.")
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

    # === LOAD MODEL ===
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
