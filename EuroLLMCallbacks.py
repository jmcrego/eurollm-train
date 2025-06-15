import os
import torch
import logging
from pathlib import Path
from sacrebleu import corpus_bleu
from transformers import TrainerCallback
from utils import greedy, run_bleu
        
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
        refs = [ref for b in self.eval_dataset for ref in self.tokenizer.batch_decode(b["references"], skip_special_tokens=True)]
        hyps = greedy(model, self.tokenizer, self.eval_dataset, max_length=self.max_length)
        bleu = run_bleu(hyps, refs, target_language=self.target_language)
        # Save file
        path = Path(f"{args.output_dir}/eval.{state.global_step}.{bleu:.2f}.out")    
        if path.parent != Path("."): # Create directory only if it's not the current one
            path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f: 
            for hyp in hyps:
                f.write(hyp + "\n")
        # Log the BLEU score into Trainer state
        logger.info(f"[Eval @ step {state.global_step}] SacreBLEU: {bleu:.2f}")
        state.log_history.append({ "eval_sacrebleu": bleu, "step": state.global_step })
        return control

def greedy2(model, tokenizer, eval_dataset, max_length, batch_size, output_file, target_language=None):
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

def get_bleu_tokenizer2(target_language):
    if target_language is None:
        return "none"
    elif target_language.lower() in {"ja", "japanese"}:
        return "ja-mecab" #requires: pip install "sacrebleu[ja]"
    elif target_language.lower() in {"zh", "chinese"}:
        return "zh" #requires pip install "sacrebleu[zh]"
    else:
        return "none"  # fallback if texts are pre-tokenized


