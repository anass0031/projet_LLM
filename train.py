# =========================
# 1. IMPORTS
# =========================
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
import torch

# =========================
# 2. DEVICE
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# =========================
# 3. LOAD DATASET (JSONL)
# =========================
dataset = load_dataset(
    "json",
    data_files="medical_dataset.jsonl"
)["train"]

# shuffle 
dataset = dataset.shuffle(seed=42)

# =========================
# 4. MODEL (Gemma)
# =========================
model_name = "google/gemma-3-1b-it"

tokenizer = AutoTokenizer.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    dtype=torch.float16
)

# =========================
# 5. LoRA CONFIG
# =========================
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# =========================
# 6. TOKENIZATION
# =========================
def tokenize(example):
    tokens = tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )
    
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

dataset = dataset.map(tokenize, batched=True)

dataset = dataset.remove_columns(
    [col for col in dataset.column_names if col not in ["input_ids", "attention_mask", "labels"]]
)

# =========================
# 7. TRAINING CONFIG
# =========================
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=3e-5,
    logging_steps=10,
    save_steps=500,
    save_total_limit=3,
    fp16=True,
    report_to="none"
)

# =========================
# 8. TRAINER
# =========================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

# =========================
# 9. TRAIN 
# =========================
print("Starting training ...")
trainer.train()

# =========================
# 10. SAVE MODEL
# =========================
model.save_pretrained("./gemma_lora_model")
tokenizer.save_pretrained("./gemma_lora_model")

print("Training finished and model saved!")
