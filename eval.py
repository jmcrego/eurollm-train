import logging
import argparse
import time
from utils import run_bleu, run_comet
# module load pytorch-gpu/py3/2.5.0

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger("training")

def read_file(path):
    with open(path, 'r', encoding='utf-8') as fd:
        return [line.strip() for line in fd]

#to download xlm-roberta-xl use:
#from transformers import AutoTokenizer, AutoModel
#AutoTokenizer.from_pretrained("facebook/xlm-roberta-xl", cache_dir="/mylocaldir/xlm-roberta-xl")
#AutoModel.from_pretrained("facebook/xlm-roberta-xl", cache_dir="/mylocaldir/xlm-roberta-xl")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to run evaluation of translation hyps using SacreBLEU and COMET.")
    parser.add_argument("-hyps", type=str, required=True, help="File with hypotheses.")
    parser.add_argument("-refs", type=str, required=True, help="File with reference.")
    parser.add_argument("-srcs", type=str, default=None, help="File with sources (not required).")
    parser.add_argument("--comet_path", type=str, default="/lustre/fsmisc/dataset/HuggingFace_Models/Unbabel/wmt23-cometkiwi-da-xl/checkpoints/model.ckpt", help="Path to COMET model.")
    parser.add_argument("--encoder_path", type=str, default=None, help="Path to xlm-roberta-xl model.")
    parser.add_argument("--target_language", type=str, default=None, help="Target language used with SacreBLEU (only if Japanese or Chinese).")
    parser.add_argument("--bleu", action='store_true', help="Run sacrebleu evaluation.")
    parser.add_argument("--comet", action='store_true', help="Run comet evaluation.")
    args = parser.parse_args()

    hyps = read_file(args.hyps)
    refs = read_file(args.refs)
    srcs = read_file(args.srcs) if args.srcs is not None else None
    assert len(hyps) == len(refs), f"Error: Different number of lines between hyps ({len(hyps)}) and refs ({len(refs)})."
    if srcs is not None:
        assert len(hyps) == len(srcs), f"Error: Different number of lines between hyps ({len(hyps)}) and srcs ({len(srcs)})."

    res = []

    if args.bleu:
        bleu = run_bleu(hyps, refs, target_language=args.target_language)
        res.append(f"{bleu:.2f}")

    if args.comet:
        comet = run_comet(hyps, refs, srcs=srcs, comet_path=args.comet_path, encoder_path=args.encoder_path)
        res.append(f"{comet:.2f}")

    res.append(args.hyps)
    print(' '.join(res))

