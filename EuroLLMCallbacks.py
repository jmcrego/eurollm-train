import os
import torch
import logging
from pathlib import Path
from sacrebleu import corpus_bleu
from transformers import TrainerCallback
from torch.utils.data import DataLoader
from EuroLLMDataset import EuroLLMDataCollator
        
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
        # Save the tokenizer to the checkpoint directory
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        self.tokenizer.save_pretrained(checkpoint_dir)
            
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
    #collator = EuroLLMDataCollator(tokenizer)
    #dataloader = DataLoader(eval_dataset, batch_size=batch_size, collate_fn=collator, shuffle=False, num_workers=1, pin_memory=True)
    device = next(model.parameters()).device
    refs = []
    hyps = []

    with torch.no_grad():
        for i, batch in enumerate(eval_dataset):
            input_ids = batch["prompt_ids"]
            labels = batch["references"]
            attention_mask = batch["prompt_mask"]
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            attention_mask = attention_mask.to(device)

            # Generate prediction (greedy decoding)
            generated_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=max_length, pad_token_id=tokenizer.pad_token_id, do_sample=False)

            for j in range(input_ids.size(0)):
                input_str = tokenizer.decode(input_ids[j], skip_special_tokens=False).replace("\n", "[\\n]")

                prediction_str = tokenizer.decode(generated_ids[j], skip_special_tokens=True)
                if "Output:\n" in prediction_str:
                    prediction_str = prediction_str.split("Output:\n", 1)[1].lstrip()
                
                cleaned_labels = [label for label in labels[j].tolist() if label != tokenizer.pad_token_id] #not needed if tokenizer.pad_token_id is removed by skip_special_tokens in next line
                reference_str = tokenizer.decode(cleaned_labels, skip_special_tokens=True)

                refs.append(reference_str)
                hyps.append(prediction_str)
                #logger.info(f"[Example {i * batch_size + j}]")
                #logger.info(f"Prompt: {input_str}")
                #logger.info(f"Ref   : {reference_str}")
                #logger.info(f"Hyp   : {prediction_str}\n")

    # BLEU computation
    bleu = 0.0
    if len(hyps) > 0:
        bleu = corpus_bleu(hyps, [refs], tokenize=get_bleu_tokenizer(target_language)).score

    # SAVE hyps
    Path(f"{output_file}.{bleu:.2f}.out").parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
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
