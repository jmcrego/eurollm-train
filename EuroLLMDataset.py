import json
import gzip
import torch
import random
import logging
import torch.nn.functional as F
from collections import defaultdict
from torch.utils.data import IterableDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger("training")


class TextFileDataset(IterableDataset):
    def __init__(self, file, loop=False):
        self._file = file
        self._loop = loop

    def __iter__(self):
        nloops = 0
        while True:
            nline = 0
            open_fn = gzip.open if self._file.endswith(".gz") else open
            mode = "rt" if self._file.endswith(".gz") else "r"
            with open_fn(self._file, mode, encoding="utf-8") as f:
                for i, line in enumerate(f):
                    nline += 1
                    yield {
                        "file": self._file,
                        "nline": nline,
                        "line": line.strip()
                    }
            nloops += 1
            logger.info(f"Exhausted file {self._file} nloops:{nloops} nlines:{nline}")
            if not self._loop:
                break


class PromptDataset(IterableDataset):
    def __init__(self, prompt, source_language, target_language, source_dataset, target_dataset):
        self._prompt = prompt
        self._source_language = source_language
        self._target_language = target_language
        self._source_dataset = source_dataset
        self._target_dataset = target_dataset

    def __iter__(self):
        prompt_template = self._prompt.replace('[source_language]', self._source_language).replace('[target_language]', self._target_language).strip()
        for source_item, target_item in zip(self._source_dataset, self._target_dataset):
            assert source_item['nline'] == target_item['nline'], f"nline numbers do not match {source_item['nline']}!={target_item['nline']}"
            yield {
                "source_file": source_item["file"],
                "target_file": target_item["file"],
                "nline": source_item["nline"],
                "source_sentence": source_item["line"], 
                "target_sentence": target_item["line"],
                "source_language": self._source_language, 
                "target_language": self._target_language, 
                "text":   prompt_template.replace('[source_sentence]', source_item["line"]).replace('[target_sentence]', target_item["line"]),
                "prompt": prompt_template.replace('[source_sentence]', source_item["line"]).replace('[target_sentence]', '')
            }

class EncodeDataset(IterableDataset):
    def __init__(self, dataset, tokenizer, mask_id=-100, is_eval=False):
        self._dataset = dataset
        self._tokenizer = tokenizer
        self._mask_id = mask_id
        self._is_eval = is_eval

    def __iter__(self):
        for e in self._dataset:
            input_ids = self._tokenize(e['text'])
            prompt_ids = self._tokenize(e['prompt'])
            # Prepend <s>, append </s> input_ids/labels manually         
            labels    = [self._mask_id]                + self._mask_labels(input_ids, prompt_ids) + [self._tokenizer.eos_token_id]
            input_ids = [self._tokenizer.bos_token_id] + input_ids                                + [self._tokenizer.eos_token_id]
            assert len(input_ids) == len(labels), f"input_ids/labels different lengths!! {len(input_ids)}!={len(labels)}"
            e["input_ids"] = input_ids                 #[<s>Translate from English into French:\nInput:\nMy father\nOutput:\nMon père</s>]
            e["labels"] = labels                       #[-100, -100, -100, -100,            ...,                       -100, Mon père</s>]
            e["attention_mask"] = [1] * len(input_ids) #all tokens are attended (unmasked), no padding yet
            if self._is_eval:
                e["prompt_ids"]  = [self._tokenizer.bos_token_id] + prompt_ids #[<s>Translate from English into French:\nInput:\nMy father\nOutput:\n]
                e["prompt_mask"] = [1] * (len(e["prompt_ids"]))                #[1, 1, 1, 1,            ...,                                        1] #all tokens are attended (no padding yet).
                if 'target_sentence' in e:
                    target_ids = self._tokenize(e['target_sentence'])
                    e["references"]  = target_ids + [self._tokenizer.eos_token_id] #[Mon père</s>]
            #self._log(e)
            yield e

    def _log(self, e):
        logger.info("")
        logger.info(f"Sample {e['nline']}")
        dec = self._detokenize(e['input_ids'])
        logger.info(f"input_ids     : {e['input_ids']} ({len(e['input_ids'])}) {dec}")
        dec = self._detokenize(e['labels'])
        logger.info(f"labels        : {e['labels']} ({len(e['labels'])}) {dec}")
        logger.info(f"attention_mask: {e['attention_mask']} ({len(e['attention_mask'])})")
        if self._is_eval:
            dec = self._detokenize(e['prompt_ids'])
            logger.info(f"prompt_ids    : {e['prompt_ids']} ({len(e['prompt_ids'])}) {dec}")
            logger.info(f"prompt_mask   : {e['prompt_mask']} ({len(e['prompt_mask'])})")
            dec = self._detokenize(e['references'])
            logger.info(f"references    : {e['references']} ({len(e['references'])}) {dec}")
            logger.info(f"target_sent   : {e['target_sentence']}")

    def _tokenize(self, txt):
        return self._tokenizer(txt, padding=False, truncation=False, add_special_tokens=False, return_offsets_mapping=False, return_tensors=None)['input_ids']                

    def _detokenize(self, ids):
        clean_ids = [id for id in ids if id != self._mask_id]  # <- filter out mask_id
        return self._tokenizer.decode(clean_ids, skip_special_tokens=False).replace('\n','⏎')

    def _mask_labels(self, input_ids, prompt_ids):
        labels = input_ids.copy()
        if len(prompt_ids) > len(input_ids):
            logger.warning(f"prompt_ids ({len(prompt_ids)}) shouldn't be larger than text_ids ({len(input_ids)})")
        labels[:len(prompt_ids)] = [self._mask_id] * len(prompt_ids)
        return labels

class FilterByLength(IterableDataset):
    def __init__(self, dataset, maximum_length):
        self._dataset = dataset
        self._maximum_length = maximum_length

    def __iter__(self):
        for e in self._dataset:
            if len(e["input_ids"]) > self._maximum_length:
                logger.info(f"filter example by length:{len(e['input_ids'])} > {self._maximum_length}")
                continue
            yield e


class BatchDataset(IterableDataset):
    def __init__ (self, dataset, shard_size=50000, batch_size=16, is_eval=False):
        self._dataset = dataset
        self._shard_size = shard_size
        self._batch_size = batch_size
        self._is_eval = is_eval
        assert shard_size % batch_size == 0, f"You better use shard_size ({shard_size}) a multiple of batch_size ({batch_size})"
        
    def __iter__(self):
        shard = []
        for e in self._dataset:
            shard.append(e)
            if len(shard) >= self._shard_size:
                yield from self._batchify(shard)
                shard = []
                
        if len(shard):
            yield from self._batchify(shard)
                
    def _batchify(self, shard):
        if not self._is_eval:
            shard = sorted(shard, key=lambda x: len(x["input_ids"]))
        batchs = []
        for i in range(0, len(shard), self._batch_size):
            batchs.append(shard[i:i+self._batch_size])
        logger.info(f"built shard with {len(batchs)} batchs from {len(shard)} samples")
        return batchs
    

class CollateDataset(IterableDataset):
    def __init__(self, dataset, tokenizer, mask_id=-100, max_avg_pad_per_sample=None):
        self._dataset = dataset
        self._tokenizer = tokenizer
        self._pad_token_id = tokenizer.pad_token_id
        self._mask_id = mask_id
        self._max_avg_pad_per_sample = max_avg_pad_per_sample

    def __iter__(self):

        for lbatch in self._dataset:
            batch = {}

            input_lens = [len(f["input_ids"]) for f in lbatch]
            input_pads = [max(input_lens) - l for l in input_lens]
            if self._max_avg_pad_per_sample is not None and sum(input_pads) > len(input_lens) * self._max_avg_pad_per_sample:
                logger.info(f"Skipping batch due to excessive padding lens: {input_lens} sum(pads): {sum(input_pads)}")
                continue

            input_ids = [torch.tensor(f["input_ids"], dtype=torch.long) for f in lbatch]
            labels = [torch.tensor(f["labels"], dtype=torch.long) for f in lbatch]
            attention_mask = [torch.tensor(f["attention_mask"], dtype=torch.long) for f in lbatch]

            padded = self._tokenizer.pad( {"input_ids": input_ids, "attention_mask": attention_mask}, padding=True, return_tensors="pt" )
            batch["input_ids"] = padded["input_ids"]
            batch["attention_mask"] = padded["attention_mask"]
        
            max_len = padded["input_ids"].shape[1]
            padded_labels = [ torch.full((max_len - len(label),), self._mask_id, dtype=torch.long) if len(label) < max_len else torch.tensor([], dtype=torch.long) for label in labels ]
            padded_labels = [ torch.cat([pad, label]) if pad.numel() > 0 else label for pad, label in zip(padded_labels, labels) ]
            padded_labels = torch.stack(padded_labels)
            batch["labels"] = padded_labels

            if "prompt_ids" in lbatch[0]: # Optional: if eval set
                prompt_ids = [torch.tensor(f["prompt_ids"], dtype=torch.long) for f in lbatch]
                prompt_mask = [torch.tensor(f["prompt_mask"], dtype=torch.long) for f in lbatch]     
                padded = self._tokenizer.pad( {"input_ids": prompt_ids, "attention_mask": prompt_mask}, padding=True, return_tensors="pt" )
                batch["prompt_ids"] = padded["input_ids"]
                batch["prompt_mask"] = padded["attention_mask"]
                if 'references' in lbatch[0]:
                    references = [torch.tensor(f["references"], dtype=torch.long) for f in lbatch]
                    padded_references = self.left_pad_sequence(references)
                    batch["references"] = padded_references

            #self._log(batch)
            yield batch

    def _log(self, batch):
        logger.info("********")
        dec = self._detokenize(batch['input_ids'][0])
        logger.info(f"input_ids     : {batch['input_ids'][0]} ({len(batch['input_ids'][0])}) {dec}")
        dec = self._detokenize(batch['labels'][0])
        logger.info(f"labels        : {batch['labels'][0]} ({len(batch['labels'][0])}) {dec}")
        logger.info(f"attention_mask: {batch['attention_mask'][0]} ({len(batch['attention_mask'][0])})")
        if "prompt_ids" in batch:
            dec = self._detokenize(batch['prompt_ids'][0])
            logger.info(f"prompt_ids    : {batch['prompt_ids'][0]} ({len(batch['prompt_ids'][0])}) {dec}")
            logger.info(f"prompt_mask   : {batch['prompt_mask'][0]} ({len(batch['prompt_mask'][0])})")
            dec = self._detokenize(batch['references'][0])
            logger.info(f"references    : {batch['references'][0]} ({len(batch['references'][0])}) {dec}")

    def _tokenize(self, txt):
        return self._tokenizer(txt, padding=False, truncation=False, add_special_tokens=False, return_offsets_mapping=False, return_tensors=None)['input_ids']                

    def _detokenize(self, ids):
        clean_ids = [id for id in ids if id != self._mask_id]  # <- filter out mask_id
        return self._tokenizer.decode(clean_ids, skip_special_tokens=False).replace('\n','⏎')


    def left_pad_sequence(self, sequences):
        reversed_seqs = [seq.flip(0) for seq in sequences]
        padded = torch.nn.utils.rnn.pad_sequence(reversed_seqs, batch_first=True, padding_value=self._pad_token_id)
        return padded.flip(dims=[1])


class MergeDatasets(IterableDataset):
    def __init__(self, datasets, weights):
        self._datasets = datasets
        self._weights = weights
        self._dataset2n = defaultdict(int)
        self._n = 0
        logger.info(f"MergeDatasets {weights}")

    def __iter__(self):
        iterators = [iter(dataset) for dataset in self._datasets]
        exhausted = [False] * len(iterators)

        while not all(exhausted):
            # Choose from only non-exhausted datasets
            available_indices = [i for i, done in enumerate(exhausted) if not done]
            available_weights = [self._weights[i] for i in available_indices]
            i = random.choices(available_indices, weights=available_weights, k=1)[0]
            try:
                logger.debug(f"Merge draws dataset {i}")
                e = next(iterators[i])
                e['dataset'] = i
                yield e
                self._n += 1
                self._dataset2n[i] += 1
                if self._n % 100000 == 0:
                    logger.info(f"Drawn {self._n} examples. Dataset counts: " + ", ".join(f"[{i}:{n}]" for i, n in sorted(self._dataset2n.items())))
            except StopIteration:
                exhausted[i] = True
                logger.info(f'Exhausted Dataset[{i}] after {self._dataset2n[i]} examples.')


class ConcatDatasets(IterableDataset):
    def __init__(self, datasets):
        self._datasets = datasets
        logger.info("ConcatDatasets")

    def __iter__(self):
        for dataset in self._datasets:
            yield from dataset
            
def create_training_dataset(
        config,
        tokenizer,
        maximum_length=512,
        shard_size=50000,
        batch_size=16,
        mask_id=-100,
        loop=False,
        max_avg_pad_per_sample=15,
):

    assert tokenizer.bos_token_id == 1, 'tokenizer.bos_token_id should be 1'
    assert tokenizer.eos_token_id == 2, 'tokenizer.eos_token_id should be 2'
    assert tokenizer.pad_token_id == 2, 'tokenizer.pad_token_id should be 2'
    assert tokenizer.padding_side == "left", 'tokenizer.padding_side should be \"left\"'

    with open(config, "r", encoding="utf-8") as f:
        configs = json.load(f)        

    logger.info(f"create_training_dataset from {config}")
    datasets = []
    weights = []
    for config in configs:
        logger.info(config)
        source_file = config.get("source_file", None)
        target_file = config.get("target_file", None)
        source_language = config.get("source_language", None)
        target_language = config.get("target_language", None)
        prompt = config.get("prompt", None)
        weight = config.get("weight", 1.0)
        weights.append(weight)

        source_dataset = TextFileDataset(source_file, loop)
        target_dataset = TextFileDataset(target_file, loop)
        dataset = PromptDataset(prompt, source_language, target_language, source_dataset, target_dataset)
        dataset = EncodeDataset(dataset, tokenizer, mask_id=mask_id, is_eval=False)
        if maximum_length:
            dataset = FilterByLength(dataset, maximum_length)
        datasets.append(dataset)
        
    if len(datasets)>1:
        dataset = MergeDatasets(datasets, weights)
    dataset = BatchDataset(dataset, shard_size, batch_size, is_eval=False)
    dataset = CollateDataset(dataset, tokenizer, mask_id=mask_id, max_avg_pad_per_sample=max_avg_pad_per_sample)
    return dataset


def create_eval_dataset(
        config,
        tokenizer,
        maximum_length=0,
        batch_size=16,
        mask_id=-100,
        max_avg_pad_per_sample=None,
):

    assert tokenizer.bos_token_id == 1, 'tokenizer.bos_token_id should be 1'
    assert tokenizer.eos_token_id == 2, 'tokenizer.eos_token_id should be 2'
    assert tokenizer.pad_token_id == 2, 'tokenizer.pad_token_id should be 2'
    assert tokenizer.padding_side == "left", 'tokenizer.padding_side should be \"left\"'

    with open(config, "r", encoding="utf-8") as f:
        configs = json.load(f)
        
    logger.info(f"create_eval_dataset from {config}")
    datasets = []
    weights = []
    for config in configs:
        logger.info(config)
        source_file = config.get("source_file", None)
        target_file = config.get("target_file", None)
        source_language = config.get("source_language", None)
        target_language = config.get("target_language", None)
        prompt = config.get("prompt", None)
        weight = config.get("weight", 1.0)
        weights.append(weight)
        source_dataset = TextFileDataset(source_file, loop=False)
        target_dataset = TextFileDataset(target_file, loop=False)
        dataset = PromptDataset(prompt, source_language, target_language, source_dataset, target_dataset)
        dataset = EncodeDataset(dataset, tokenizer, mask_id=mask_id, is_eval=True)
        if maximum_length:
            dataset = FilterByLength(dataset, maximum_length)
        datasets.append(dataset)
        
    if len(datasets)>1:
        dataset = ConcatDatasets(datasets)
    dataset = BatchDataset(dataset, batch_size=batch_size, is_eval=True)
    dataset = CollateDataset(dataset, tokenizer, mask_id=mask_id, max_avg_pad_per_sample=max_avg_pad_per_sample)
    return dataset


def create_test_dataset(
    file_path,
    source_language,
    target_language,
    prompt,
    tokenizer,
    batch_size=16,
    mask_id=-100,
):

    assert tokenizer.bos_token_id == 1, 'tokenizer.bos_token_id should be 1'
    assert tokenizer.eos_token_id == 2, 'tokenizer.eos_token_id should be 2'
    assert tokenizer.pad_token_id == 2, 'tokenizer.pad_token_id should be 2'
    assert tokenizer.padding_side == "left", 'tokenizer.padding_side should be \"left\"'

    prompt_template = prompt.replace('[source_language]', source_language).replace('[target_language]', target_language).strip()

    dataset = []
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            e = {
#                'source_language': source_language,
#                'target_language': target_language,
#                'source_sentence': line.strip(),
#                'nline': i+1,
                "prompt": prompt_template.replace('[source_sentence]', line.strip()).replace('[target_sentence]', ''),
                "text":   prompt_template.replace('[source_sentence]', line.strip()), ### not to be used for testing ([target_sentence] remains in prompt)
            }
            dataset.append(e)

    dataset = EncodeDataset(dataset, tokenizer, mask_id=mask_id, is_eval=True)
    dataset = BatchDataset(dataset, batch_size=batch_size, is_eval=True)
    dataset = CollateDataset(dataset, tokenizer, mask_id=mask_id)
    return dataset



if __name__ == "__main__":
    from transformers import AutoTokenizer

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    logger = logging.getLogger("training")

    tokenizer = AutoTokenizer.from_pretrained("/lustre/fsmisc/dataset/HuggingFace_Models/utter-project/EuroLLM-1.7B")
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = create_training_dataset(
        config="data/train/train.en-ja.Misc__HiqhQuality.json",
        tokenizer=tokenizer,
        maximum_length=512,
        shard_size=batch_size*1000, #use a multiple of batch_size
        batch_size=batch_size,
        is_eval=False,
        loop=False,
    )

    from torch.utils.data import DataLoader    
    dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size, 
        num_workers=1,
        pin_memory=True,
        timeout=10
    )
    
    for i, batch in enumerate(dataloader):
        continue
        n_padded = (batch['input_ids'] == tokenizer.pad_token_id).sum().item() - batch['input_ids'].shape[0]
        print(f"\nBatch {i} shape: {batch['input_ids'].shape} n_padded_tokens = {n_padded}")
        for j in range(batch['input_ids'].shape[0]):
            print(f"Batch {i} Example {j}")
            
            input_ids = batch["input_ids"][j].tolist()
            dec = tokenizer.decode(input_ids, skip_special_tokens=False)#.replace('\n','⏎')
            print(f"decode(input_ids) -->{dec}<--")

            #label_ids = batch["labels"][j].tolist()
            #dec = tokenizer.decode([t for t in label_ids if t != -100], skip_special_tokens=False)
            #print(f"decode(label_ids) -->{dec}<--")

            #list_of_input_ids_str = [f"{id_}:{tokenizer.convert_ids_to_tokens(id_)}" for id_ in input_ids]
            #print(f"input_ids -->{list_of_input_ids_str}<--")

            #list_of_label_ids_str = [f"{id_}:{tokenizer.convert_ids_to_tokens(id_)if id_>=0 else id_}" for id_ in label_ids]
            #print(f"label_ids -->{list_of_label_ids_str}<--")

            #attention_mask = batch["attention_mask"][0].tolist()            
            #print(f"attention_mask -->{attention_mask}<--")

        
