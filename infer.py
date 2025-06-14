import logging
import argparse
import torch
import time
from utils import run_bleu, run_comet, greedy

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger("training")

def get_dtype(args):
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
    return dtype

def run_vllm(args):
    from vllm import LLM, SamplingParams
    dtype = get_dtype(args)

    # === LOAD MODEL ===
    llm = LLM(model=args.m, tokenizer=args.t if args.t is not None else args.m, trust_remote_code=True, dtype=dtype)

    # === LOAD DATASET ===
    prompts = []
    with open(args.i, 'r') as fd:
        for line in fd:
            prompt = args.p.replace('[source_language]', args.sl).replace('[target_language]', args.tl).replace('[source_sentence]', line.strip())
            prompts.append(prompt)

    # ===INFERENCE ===
    sampling_params_greedy = SamplingParams( temperature=0.0, max_tokens=args.max_tokens, stop=["\n", "</s>"] )
    #sampling_params_beams5 = SamplingParams(use_beam_search=True, max_tokens=args.max_tokens, num_beams=5, early_stopping=True, stop=["\n", "</s>"])
    #sampling_params_random = SamplingParams( temperature=0.7, top_p=0.9, top_k=50, max_tokens=args.max_tokens, stop=["\n", "</s>"], seed=42)
    outputs = llm.generate(prompts, sampling_params_greedy) # Run inference
    
    hyps = []
    for i in range(len(prompts)):
        prompt = prompts[i].replace("\n","⏎")
        output = outputs[i].outputs[0].text.strip().replace("\n","⏎")
        hyps.append(output)
        logger.info(f"Sample {i+1}")
        logger.info(f"Prompt: {prompt}")
        logger.info(f"Output: {output}")
        logger.info("----------------------------------------")
    return hyps


def run_hf(args):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from EuroLLMDataset import create_test_dataset
    dtype = get_dtype(args)

    # === LOAD TOKENIZER ===
    tokenizer = AutoTokenizer.from_pretrained(args.t if args.t is not None else args.m, use_fast=True)
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token
    logger.info(f"Loaded tokenizer {args.t if args.t is not None else args.m}")

    # === LOAD MODEL ===
    model = AutoModelForCausalLM.from_pretrained(args.m, torch_dtype=dtype, device_map="auto")
    model.config.pad_token_id = tokenizer.pad_token_id
    logger.info(f"Loaded model from pretrained {args.m}")

    # === LOAD DATASET ===
    dataset = create_test_dataset(args.i, args.sl, args.tl, args.p, tokenizer, batch_size=args.batch_size)

    # === INFERENCE ===
    hyps = greedy(model, tokenizer, dataset, args.max_tokens)
    return hyps


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to run inference of EuroLLM models using vLLM.")
    parser.add_argument("engine", type=str, help="Engine to perform inference, use: vllm OR hf.")
    parser.add_argument("-m", type=str, default="/lustre/fsmisc/dataset/HuggingFace_Models/utter-project/EuroLLM-1.7B", help="Path where the EuroLLM model is found.")
    parser.add_argument("-t", type=str, default=None, help="Path where the EuroLLM tokenizer is found (if different to -m).")
    parser.add_argument("-i", type=str, required=True, help="Input file path.")
    parser.add_argument("-o", type=str, default=None, help="Output file path.")
    parser.add_argument("-r", type=str, default=None, help="Reference file path.")
    parser.add_argument("-sl", type=str, required=True, help="Source language.")
    parser.add_argument("-tl", type=str, required=True, help="Target language.")
    parser.add_argument("--p", type=str, default="Translate from [source_language] to [target_language]:\nInput:\n[source_sentence]\nOutput:\n", help="Prompt used.")
    parser.add_argument("--bf16", action='store_true', help="Use torch_dtype=torch.bfloat16.")
    parser.add_argument("--fp16", action='store_true', help="Use torch_dtype=torch.float16.")
    parser.add_argument("--max_tokens", type=int, default=512, help="Max sequence length.")
    parser.add_argument("--batch_size", type=int, default=16, help="Per device batch size.")
    parser.add_argument("--bleu", action='store_true', help="Run sacrebleu evaluation.")
    parser.add_argument("--comet", action='store_true', help="Run comet evaluation.")
    args = parser.parse_args()

    tic = time.time()
    if args.engine.lower() == 'vllm':
        hyps = run_vllm(args)
    elif args.engine.lower() == 'hf':
        hyps = run_hf(args)
    else:
        raise ValueError("engine value should be either: 'vllm' or 'hf'")
    logger.info(f"Inference took {time.time()-tic:.2f} seconds with engine {args.engine}")

    # === EVAL ===
    bleu = 0.0
    comet = 0.0
    if args.r is not None:
        refs = []
        with open(args.r, 'r') as fd:
            for line in fd:
                refs.append(line.strip())

        bleu = run_bleu(hyps, refs, target_language=args.tl) if args.bleu else 0.0
        comet = run_comet(hyps, refs) if args.comet else 0.0

    # === OUTPUT ===
    if args.o is not None:
        args.o = args.o.replace('[BLEU]', f"{bleu:.2f}").replace('[COMET]', f"{comet:.2f}")
        with open(args.o, 'w') as fd:
            for h in hyps:
                fd.write(h + '\n')
        logger.info(f"Hypotheses saved in: {args.o}")
