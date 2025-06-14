from vllm import LLM, SamplingParams
import logging
import argparse
import sacrebleu
import torch
import json
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger("training")

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
    parser = argparse.ArgumentParser(description="Script to run inference of EuroLLM models using vLLM.")
    parser.add_argument("-m", type=str, default="/lustre/fsmisc/dataset/HuggingFace_Models/utter-project/EuroLLM-1.7B", help="Path where the EuroLLM model is found.")
    parser.add_argument("-t", type=str, default=None, help="Path where the EuroLLM tokenizer is found (if different to -m).")
    parser.add_argument("-i", type=str, required=True, help="Input file path.")
    parser.add_argument("-o", type=str, default=None, help="Output file path.")
    parser.add_argument("-r", type=str, default=None, help="Reference file path.")
    parser.add_argument("-sl", type=str, required=True, help="Source language.")
    parser.add_argument("-tl", type=str, required=True, help="Target language.")
    parser.add_argument("--max_tokens", type=int, default=512, help="Max sequence length.")
    parser.add_argument("--batch_size", type=int, default=16, help="Per device batch size.")
    parser.add_argument("--bleu", action='store_true', help="Run sacrebleu evaluation.")
    parser.add_argument("--comet", action='store_true', help="Run comet evaluation.")
    args = parser.parse_args()

    # Initialize vLLM with your local model
    llm = LLM(model=args.m, tokenizer=args.t if args.t is None else args.m, trust_remote_code=True, dtype="bfloat16")  # or "float16" if needed

    prompts = []
    with open(args.i, 'r') as fd:
        for line in fd:
            prompts.append(f"Translate from {args.sl} to {args.tl}:\nInput:\n{line.strip()}\nOutput:\n")

    refs = []
    if args.r is not None:
        with open(args.r, 'r') as fd:
            for line in fd:
                refs.append(line.strip())


    #sampling_params_greedy = SamplingParams( temperature=0.0, max_tokens=args.max_tokens, stop=["\n", "</s>"] )
    #sampling_params_beams5 = SamplingParams(use_beam_search=True, max_tokens=args.max_tokens, num_beams=5, early_stopping=True, stop=["\n", "</s>"])
    sampling_params_random = SamplingParams( temperature=0.7, top_p=0.9, top_k=50, max_tokens=args.max_tokens, stop=["\n", "</s>"], seed=42)
    outputs = llm.generate(prompts, sampling_params_random) # Run inference
    
    hyps = []
    for i in range(len(prompts)):
        prompt = prompts[i].replace("\n","\\n")
        output = outputs[i].outputs[0].text.strip()
        hyps.append(output)
        print(f"Prompt: {prompt}")
        print(f"Output: {output}")
        if len(refs):
            ref = refs[i]
            print(f"Ref   : {ref}")
        print("-" * 40)

    if args.o is not None:
        with open(args.o, 'w') as fd:
            for h in hyps:
                fd.write(h + '\n')
        
    if args.bleu and len(refs):
        assert len(hyps) == len(refs), "Number of hypotheses and references must match."
        bleu = sacrebleu.corpus_bleu(hyps, [refs], tokenize=get_bleu_tokenizer(args.tl))
        print(f"SacreBLEU score: {bleu.score:.2f} {args.m}")

