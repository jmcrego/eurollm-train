import os
import torch
import logging
#import argparse
#from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
#from EuroLLMDataset import create_test_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger("training")

def run_comet(hyps, refs, srcs=None, comet_path="/lustre/fsmisc/dataset/HuggingFace_Models/Unbabel/wmt23-cometkiwi-da-xl/checkpoints/model.ckpt"):
    from comet import load_from_checkpoint
    data = []
    for i in range(len(hyps)):
        data.append({ "src": srcs[i] if srcs else "", "mt": hyps[i], "ref": refs[i] })
    m = load_from_checkpoint(comet_path)
    scores = m.predict(data, batch_size=8, gpus=1 if m.hparams.use_gpu else 0)
    logger.info(f"COMET scores: {scores}")
    return scores['mean']

def run_bleu(hyps, refs, target_language=None):
    import sacrebleu
    def get_bleu_tokenizer(target_language):
        if target_language is None:
            return "none"
        elif target_language.lower() in {"ja", "japanese"}:
            return "ja-mecab" #requires: pip install "sacrebleu[ja]"
        elif target_language.lower() in {"zh", "chinese"}:
            return "zh" #requires pip install "sacrebleu[zh]"
        else:
            return "none"  # fallback if texts are pre-tokenized

    assert len(hyps) == len(refs), "Number of hypotheses and references must match."
    bleu = sacrebleu.corpus_bleu(hyps, [refs], tokenize=get_bleu_tokenizer(target_language)).score
    logger.info(f"SacreBLEU score: {bleu:.2f}")
    return bleu


def greedy(model, tokenizer, dataset, max_length):
    model.eval()
    device = next(model.parameters()).device

    hyps = []
    with torch.no_grad():
        n = 0
        for i, batch in enumerate(dataset):
            batch = {k: v.to(device) for k, v in batch.items()}
            input_ids = batch["prompt_ids"]
            attention_mask = batch["prompt_mask"]

            attention_mask = attention_mask.to(device)
            input_ids = input_ids.to(device)

            generated_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=max_length, pad_token_id=tokenizer.pad_token_id, do_sample=False)

            prompt_len = len(input_ids[0]) ### all prompts have same length (left-padded)
            for j in range(input_ids.size(0)):
                n += 1
                input_str = tokenizer.decode(input_ids[j], skip_special_tokens=True).replace("\n", "⏎")
                prediction_str = tokenizer.decode(generated_ids[j][prompt_len:], skip_special_tokens=True).replace("\n", "⏎") #remove prompt
                hyps.append(prediction_str)
                logger.info(f"Sample {n}")
                logger.info(f"Prompt: {input_str}")
                logger.info(f"Output: {prediction_str}")
                logger.info("----------------------------------------")

    return hyps
