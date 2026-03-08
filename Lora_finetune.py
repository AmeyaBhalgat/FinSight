import os
import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

BASE_MODEL    = "meta-llama/Llama-2-7b-hf"
OUTPUT_DIR    = "lora_adapter"
TRAINING_DATA = "finetune_data.jsonl"

# ── Training data format (finetune_data.jsonl) ────────────────────────────────
# Each line is one JSON object:
# {
#   "question": "What is Apple's revenue growth?",
#   "context":  "Apple reported Q3 revenue of $81.8B...",
#   "answer":   "Revenue grew 8% to $81.8B [Source: apple_q3.pdf, Page: 3]"
# }


def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        load_in_8bit=True,
        device_map="auto",
        torch_dtype=torch.float16
    )
    return model, tokenizer


def apply_lora(model):
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none"
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


def prepare_dataset(tokenizer):
    examples = []
    with open(TRAINING_DATA, "r") as f:
        for line in f:
            item = json.loads(line.strip())
            text = (
                f"Answer using only the context below. Cite every fact as [Source: filename, Page: N].\n\n"
                f"Context: {item['context']}\n\n"
                f"Question: {item['question']}\n\n"
                f"Answer: {item['answer']}"
            )
            examples.append({"text": text})

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=1024, padding="max_length")

    return Dataset.from_list(examples).map(tokenize, batched=True, remove_columns=["text"])


def train(model, tokenizer, dataset):
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        report_to="none"
    )
    Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    ).train()

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"LoRA adapter saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    model, tokenizer = load_model_and_tokenizer()
    model = apply_lora(model)
    dataset = prepare_dataset(tokenizer)
    train(model, tokenizer, dataset)