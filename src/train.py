import torch
from datasets import load_dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
)

from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

# =====================================================
# CONFIG
# =====================================================

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

DATASET_PATH = "dataset/train.jsonl"

OUTPUT_DIR = "outputs"

# =====================================================
# LOAD DATASET
# =====================================================

print("\nLoading dataset...\n")

dataset = load_dataset(
    "json",
    data_files=DATASET_PATH
)["train"]

print(f"Dataset Loaded: {len(dataset)} samples")

# =====================================================
# TOKENIZER
# =====================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

tokenizer.pad_token = tokenizer.eos_token

# =====================================================
# FORMAT DATA (IMPORTANT FIX)
# =====================================================

def format_example(example):
    return {
        "text": tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False
        )
    }

dataset = dataset.map(format_example)

print("\nSample check:\n", dataset[0]["text"][:300])

# =====================================================
# MODEL
# =====================================================

print("\nLoading model...\n")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

# =====================================================
# LORA CONFIG
# =====================================================

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ],
)

model = get_peft_model(model, lora_config)

model.print_trainable_parameters()

# =====================================================
# TRAINING ARGUMENTS
# =====================================================

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    learning_rate=2e-4,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    logging_steps=1,
    save_strategy="epoch",
    report_to="none",
    fp16=True,
    optim="adamw_torch",
)

# =====================================================
# TRAINER
# =====================================================

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

# =====================================================
# TRAIN
# =====================================================

print("\nStarting training...\n")

trainer.train()

print("\nTraining completed!")

# =====================================================
# SAVE MODEL
# =====================================================

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("\nModel saved successfully at:", OUTPUT_DIR)