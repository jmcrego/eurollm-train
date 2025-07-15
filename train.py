import os
import torch
import logging
import argparse
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from EuroLLMDataset import create_training_dataset, create_eval_dataset
from EuroLLMCallbacks import CustomBLEUCallback, LogStepCallback, SaveTokenizerCallback, SaveBestBLEUCheckpoints
from torch.utils.data import DataLoader
from transformers.trainer_utils import get_last_checkpoint
from utils import set_seed

#transformers.logging.set_verbosity_warning() #This will suppress their internal INFO level logs (only LogStepCallback is used)
os.environ["TOKENIZERS_PARALLELISM"] = "false" #prevents a warning
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger("training")

class CustomTrainer(Trainer):
    def __init__(self, *args, train_dataloader=None, eval_dataloader=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._train_dataloader = train_dataloader
        self._eval_dataloader = eval_dataloader

    def get_train_dataloader(self):
        if self._train_dataloader is not None:
            return self._train_dataloader
        return super().get_train_dataloader()

    def get_eval_dataloader(self, eval_dataset=None):
        if self._eval_dataloader is not None:
            return self._eval_dataloader
        return super().get_eval_dataloader(eval_dataset)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to continue Pre-Training EuroLLM models.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model_path", type=str, default="/lustre/fsmisc/dataset/HuggingFace_Models/utter-project/EuroLLM-1.7B", help="Path to the EuroLLM model.")
    parser.add_argument("--save_path", type=str, default="./EuroLLM-1.7B-CPT", help="Path where the model will be saved.")
    parser.add_argument("--train_config", type=str, default="data/train/train.json", help="Training json config file.")
    parser.add_argument("--eval_config", type=str, default="data/eval/eval.json", help="Eval json config file.")
    parser.add_argument("--max_length", type=int, default=512, help="Max sequence length.")
    parser.add_argument("--per_device_batch_size", type=int, default=16, help="Per device batch size.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps.")
    parser.add_argument("--max_steps", type=int, default=50000, help="Number of model updates.")
    parser.add_argument("--logging_steps", type=int, default=100, help="Logging every this many steps.")
    parser.add_argument("--eval_steps", type=int, default=1000, help="Run evaluation every this many steps.")
    parser.add_argument("--save_steps", type=int, default=1000, help="Save checkpoint every this many steps.")
    parser.add_argument("--save_total_limit", type=int, default=5, help="Max number of checkpoints saved.")
    parser.add_argument("--learning_rate", type=float, default=7e-6, help="Learning rate (use 5e-5 if lora).")
    parser.add_argument("--shard_size", type=int, default=64000, help="Number of lines in a shard.")
    parser.add_argument("--mask_id", type=int, default=-100, help="Id used to mask the prompt when learning (compute loss over target only).")
    parser.add_argument("--target_language_sacrebleu", type=str, default=None, help="target language for sacrebleu string tokenizer.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for randomness.")
    parser.add_argument("--loop", action='store_true', help="Loop on training dataset/s.")
    parser.add_argument("--grad_ckpt", action='store_true', help="Enable gradient checkpointing")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Clip gradients to max_grad_norm")
    parser.add_argument("--lora", action='store_true', help="Use low-rank adaptors for training.")
    parser.add_argument("--bf16", action='store_true', help="Use torch_dtype=torch.bfloat16.")
    parser.add_argument("--fp16", action='store_true', help="Use torch_dtype=torch.float16.")
    args = parser.parse_args()

    if args.seed:
        set_seed(args.seed)

    if args.lora:
        from peft import get_peft_model, LoraConfig, TaskType
        logger.info("Running LoRA training")
    else:
        logger.info("Running full training")

    # === LOAD TOKENIZER ===
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token
    logger.info(f"Loaded tokenizer {args.model_path}")
    logger.info(f"Tokenizer BOS token id: {tokenizer.bos_token_id}: {tokenizer.bos_token}")
    logger.info(f"Tokenizer EOS token id: {tokenizer.eos_token_id}: {tokenizer.eos_token}")
    logger.info(f"Tokenizer PAD token id: {tokenizer.pad_token_id}: {tokenizer.pad_token}")
    logger.info(f"Tokenizer padding_side: {tokenizer.padding_side}")

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

    #Clear cache before model loading
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=dtype)
    model.config.pad_token_id = tokenizer.pad_token_id
    logger.info(f"Loaded model from pretrained {args.model_path}")
    logger.info(f"Model dtype={next(model.parameters()).dtype}")
    logger.info(f"Model config BOS token id: {model.config.bos_token_id}")
    logger.info(f"Model config PAD token id: {model.config.pad_token_id}")

    if args.lora:
        lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.1, bias="none", task_type=TaskType.CAUSAL_LM) 
        model = get_peft_model(model, lora_config) 
        model.print_trainable_parameters()  #Optional: logs trainable param count

    if args.grad_ckpt:
        logger.info(f"Enable gradient checkpointing")
        model.config.use_cache = False
        model.gradient_checkpointing_enable()

    # === LOAD DATASETS ===
    train_dataset = create_training_dataset(args.train_config, tokenizer, maximum_length=args.max_length, shard_size=args.shard_size, batch_size=args.per_device_batch_size, mask_id=args.mask_id, loop=args.loop)
    eval_dataset = create_eval_dataset(args.eval_config, tokenizer, batch_size=args.per_device_batch_size, mask_id=args.mask_id)
    logger.info("Loaded iterable datasets")

    # === TRAINING ARGS ===
    training_args = TrainingArguments(
        output_dir=args.save_path,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=max(1, args.per_device_batch_size // 2),
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        max_grad_norm=args.max_grad_norm,
        # Evaluation settings (must match save settings)
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        # Save settings (must match eval settings)
        save_strategy="steps",
        save_steps=args.eval_steps,  # Same as eval_steps
        save_total_limit=1, #use minimum (we'll handle it ourselves via SaveBestBLEUCheckpoints callback)
        # BLEU-based model selection
        load_best_model_at_end=True,
        metric_for_best_model="eval_sacrebleu",
        greater_is_better=True,
        # Other settings
        logging_dir=os.path.join(args.save_path, "logs"),
        logging_steps=args.logging_steps,
        bf16=args.bf16,
        fp16=args.fp16,
        disable_tqdm=True,
        ignore_data_skip=True, #to allow raising Value in data collator (it prevents from skipping batchs when fine tuning after n iterations)
        remove_unused_columns=False,
        report_to="none", #"tensorboard",
    )

    torch.cuda.synchronize()
    logger.info("CUDA sync passed")

    # === DATA LOADER ===
    # batch_size=None is VERY IMPORTANT: disables automatic batching 
    # collate_fn indicates that EuroLLMDataCollator returns full batches
    train_loader = DataLoader(dataset=train_dataset, batch_size=None, collate_fn=None)
    eval_loader = DataLoader(dataset=eval_dataset, batch_size=None, collate_fn=None)

    # === CALLBACKS ===
    log_callback = LogStepCallback()
    bleu_callback = CustomBLEUCallback(tokenizer, eval_dataset, max_length=args.max_length, batch_size=args.per_device_batch_size, target_language=args.target_language_sacrebleu)
    save_tokenizer_callback = SaveTokenizerCallback(tokenizer)
    bleu_checkpoint_callback = SaveBestBLEUCheckpoints(save_total_limit=args.save_total_limit)

    # === INIT TRAINER ===
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=[{}],  # dummy just to pass validation
        eval_dataset=[{}],   # dummy just to pass validation
        data_collator=None,  # not needed, already collated from datasets
        train_dataloader=train_loader,
        eval_dataloader=eval_loader,
        callbacks=[log_callback, bleu_callback, save_tokenizer_callback, bleu_checkpoint_callback],
    )

    resume_checkpoint_path = get_last_checkpoint(args.save_path)
    if resume_checkpoint_path:
        logger.info(f"Resuming training from {resume_checkpoint_path}")
    else:
        logger.info("Starting training from scratch")

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i}:")
            logger.info(f"  Allocated: {torch.cuda.memory_allocated(i) / 1024**2:.2f} MB")
            logger.info(f"  Reserved : {torch.cuda.memory_reserved(i) / 1024**2:.2f} MB")
            logger.info(f"  Free     : {(torch.cuda.memory_reserved(i) - torch.cuda.memory_allocated(i)) / 1024**2:.2f} MB")


    trainer.train(resume_from_checkpoint=resume_checkpoint_path if resume_checkpoint_path else None)
    trainer.evaluate()
    if trainer.is_world_process_zero():
        model.save_pretrained(os.path.join(args.save_path, "final"))
        tokenizer.save_pretrained(os.path.join(args.save_path, "final"))

    if args.lora:
        model.merge_and_unload() # save full (merged) model
        model.save_pretrained(os.path.join(args.save_path, "merged_model"))

    logger.info("Done")

