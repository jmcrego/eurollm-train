import os
import torch
import logging
import shutil
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

        # Log the BLEU score in two places: in log_history (for your tracking), in metrics (for Trainer's best model tracking)
        if not hasattr(state, 'log_history'):
            state.log_history = []
        state.log_history.append({"eval_sacrebleu": bleu, "step": state.global_step})
        if metrics := kwargs.get('metrics', None):
            metrics['eval_sacrebleu'] = bleu
        
        logger.info(f"[Eval @ step {state.global_step}] SacreBLEU: {bleu:.2f}")
        return control

class SaveBestBLEUCheckpoints(TrainerCallback):
    def __init__(self, save_total_limit=5):
        super().__init__()
        self.save_total_limit = save_total_limit
        
    def get_checkpoints(self, state, output_dir):
        """Returns sorted list of (bleu_score, path, step) for existing checkpoints"""
        checkpoints = []
        output_path = Path(output_dir)
    
        # First get all existing checkpoint steps
        existing_steps = set()
        for item in output_path.iterdir():
            if item.is_dir() and item.name.startswith('checkpoint-'):
                try:
                    step = int(item.name.split('-')[1])
                    existing_steps.add(step)
                except (ValueError, IndexError):
                    continue
    
        # Then match with log_history
        for entry in reversed(state.log_history):
            if 'eval_sacrebleu' in entry and 'step' in entry:
                step = entry['step']
                if step in existing_steps:
                    checkpoints.append((entry['eval_sacrebleu'], str(output_path / f"checkpoint-{step}"), step))
    
        return sorted(checkpoints, key=lambda x: (-x[0], x[2]))

    def on_step_end(self, args, state, control, **kwargs):
        # Only run at eval steps
        if state.global_step % args.eval_steps != 0:
            return
        checkpoints = self.get_checkpoints(state, args.output_dir)
        logger.info(f"Available checkpoints: {checkpoints}")
        current_path = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        logger.info(f"Preparing to save checkpoint: {current_path}")
        # Always allow the Trainer to save (we'll clean up afterwards)
        control.should_save = True
        # Clean up old checkpoints if needed
        if len(checkpoints) >= self.save_total_limit:
            # Keep top N-1 plus the new one we're about to save
            for _, path, _ in checkpoints[self.save_total_limit-1:]:
                if Path(path).exists():
                    shutil.rmtree(path)
                    logger.info(f"Removed old checkpoint {path} to maintain limit")
        return control




