from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import Dataset


class LLMDataset(Dataset):
    def __init__(self, tokenized_data):
        self.data = tokenized_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "input_ids": item["input_ids"],
            "attention_mask": item["attention_mask"],
            "labels": item["input_ids"],  # CLM: input과 동일
        }


def build_dataset(config, tokenizer_name="gpt2", dataset_name="wikitext",
                  dataset_config="wikitext-2-raw-v1"):
    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token

    # 데이터셋 로드
    raw_dataset = load_dataset(dataset_name, dataset_config)

    # 짧은 문장들을 이어붙여서 max_seq_len 단위로 자르는 방식
    # 단순 truncation보다 데이터 낭비가 적음
    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=False,       # 일단 자르지 않음
            padding=False,
            return_attention_mask=False,
        )

    def group_texts(examples):
        # 모든 토큰을 이어붙임
        concatenated = sum(examples["input_ids"], [])
        total_len = (len(concatenated) // config.max_seq_len) * config.max_seq_len

        # max_seq_len 단위로 자름
        chunks = [
            concatenated[i : i + config.max_seq_len]
            for i in range(0, total_len, config.max_seq_len)
        ]
        attention_masks = [[1] * config.max_seq_len for _ in chunks]

        return {
            "input_ids": chunks,
            "attention_mask": attention_masks,
        }

    # 빈 문장 제거
    raw_dataset = raw_dataset.filter(lambda x: len(x["text"].strip()) > 0)

    tokenized = raw_dataset.map(
        tokenize,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing",
    )

    grouped = tokenized.map(
        group_texts,
        batched=True,
        desc="Grouping into chunks",
    )

    train_dataset = LLMDataset(grouped["train"])
    valid_dataset = LLMDataset(grouped["validation"])

    return train_dataset, valid_dataset, tokenizer