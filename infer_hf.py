import os
import torch
import logging
import argparse
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from EuroLLMDataset import create_test_dataset
import sacrebleu
from comet import load_from_checkpoint

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger("training")

def run_comet(hyps, refs, srcs=None, comet_path="/lustre/fsmisc/dataset/HuggingFace_Models/Unbabel/wmt23-cometkiwi-da-xl/checkpoints/model.ckpt"):
    data = []
    for i in range(len(hyps)):
        data.append({ "src": srcs[i] if srcs else "", "mt": hyps[i], "ref": refs[i] })
    m = load_from_checkpoint(comet_path)
    scores = m.predict(data, batch_size=8, gpus=1 if m.hparams.use_gpu else 0)
    logger.info(f"COMET scores: {scores}")
    return scores['mean']

def get_bleu_tokenizer(target_language):
    if target_language is None:
        return "none"
    elif target_language.lower() in {"ja", "japanese"}:
        return "ja-mecab" #requires: pip install "sacrebleu[ja]"
    elif target_language.lower() in {"zh", "chinese"}:
        return "zh" #requires pip install "sacrebleu[zh]"
    else:
        return "none"  # fallback if texts are pre-tokenized


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
                logger.info(f"Hyp   : {prediction_str}")
                logger.info("----------------------------------------")

    return hyps

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to continue Pre-Training EuroLLM models.")
    parser.add_argument("-m", type=str, default="/lustre/fsmisc/dataset/HuggingFace_Models/utter-project/EuroLLM-1.7B", help="Path where the EuroLLM model is found.")
    parser.add_argument("-t", type=str, default=None, help="Path where the EuroLLM tokenizer is found (if different to -m).")
    parser.add_argument("-i", type=str, required=True, help="Input file path.")
    parser.add_argument("-o", type=str, default=None, help="Output file path.")
    parser.add_argument("-r", type=str, default=None, help="Reference file path.")
    parser.add_argument("-sl", type=str, required=True, help="Source language.")
    parser.add_argument("-tl", type=str, required=True, help="Target language.")
    parser.add_argument("--p", type=str, default="Translate from [source_language] to [target_language]:\nInput:\n[source_sentence]\nOutput:\n[target_sentence]", help="Prompt used.")
    parser.add_argument("--bf16", action='store_true', help="Use torch_dtype=torch.bfloat16.")
    parser.add_argument("--fp16", action='store_true', help="Use torch_dtype=torch.float16.")
    parser.add_argument("--max_tokens", type=int, default=512, help="Max sequence length.")
    parser.add_argument("--batch_size", type=int, default=16, help="Per device batch size.")
    parser.add_argument("--bleu", action='store_true', help="Run sacrebleu evaluation.")
    parser.add_argument("--comet", action='store_true', help="Run comet evaluation.")
    args = parser.parse_args()

    # === LOAD TOKENIZER ===
    tokenizer = AutoTokenizer.from_pretrained(args.t if args.t is not None else args.m, use_fast=True)
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token
    logger.info(f"Loaded tokenizer {args.t if args.t is not None else args.m}")

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
    model = AutoModelForCausalLM.from_pretrained(args.m, torch_dtype=dtype, device_map="auto")
    model.config.pad_token_id = tokenizer.pad_token_id
    logger.info(f"Loaded model from pretrained {args.m}")

    # === LOAD DATASET ===
    dataset = create_test_dataset(args.i, args.sl, args.tl, args.p, tokenizer, batch_size=args.batch_size)

    # === INFERENCE ===
    hyps = greedy(model, tokenizer, dataset, args.max_tokens)

    # === EVAL ===
    refs = []
    if args.r is not None:
        with open(args.r, 'r') as fd:
            for line in fd:
                refs.append(line.strip())

    bleu = 0.0
    if args.bleu and len(refs):
        assert len(hyps) == len(refs), "Number of hypotheses and references must match."
        bleu = sacrebleu.corpus_bleu(hyps, [refs], tokenize=get_bleu_tokenizer(args.tl)).score
        logger.info(f"SacreBLEU score: {bleu:.2f} {args.m}")

    comet = 0.0
    if args.comet and len(refs):
        assert len(hyps) == len(refs), "Number of hypotheses and references must match."
        comet = 0.        
        logger.info(f"COMET score: {comet:.2f} {args.m}")

    # === OUTPUT ===
    if args.o is not None:
        args.o = args.o.replace('[BLEU]', f"{bleu:.2f}").replace('[COMET]', f"{comet:.2f}")
        with open(args.o, 'w') as fd:
            for h in hyps:
                fd.write(h + '\n')
        logger.info(f"Hypotheses saved in: {args.o}")
