from transformers import Trainer, TrainingArguments, AutoTokenizer
from datasets import load_dataset
from model.llm import LLM
from config import ModelConfig
from utils.lora_utils import freeze_non_lora, print_trainable_prams
from data.dataset import build_dataset

config = ModelConfig()
model = LLM(config)
freeze_non_lora(model)
print_trainable_prams(model)

train_dataset, valid_dataset, tokenizer = build_dataset(config)

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
def tokenize(examples):
    return tokenizer(examples["text"], truncation=True,
                     max_length=config.max_seq_len, padding="max_length")
tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
tokenized = tokenized.map(lambda x: {"labels": x["input_ids"]})

training_args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=8,
    num_train_epochs=3,
    learning_rate=3e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    fp16=True,
    logging_steps=100,
    save_strategy="epoch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
)
trainer.train()