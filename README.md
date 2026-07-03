# 🔮 Vedaz AI - Vedic Astrology Fine-Tuning Assignment

This repository contains my solution for the **Vedaz AI Internship Technical Assignment**, where I fine-tuned a Large Language Model (LLM) on Vedic astrology conversations using **LoRA (Low-Rank Adaptation)**.

---

## 📌 Assignment Objectives

### ✅ Technical Task
- Fine-tune an LLM on the provided astrology conversations.
- Use parameter-efficient fine-tuning (LoRA).
- Prepare the dataset in chat format.
- Save the trained adapter for inference.

### ✅ Documentation Task
- Write the process of hosting the model on a VPS using **vLLM**.
- Create five manually written astrology conversations for training.

---

# 🛠️ Tech Stack

- Python 3.12
- Hugging Face Transformers
- PEFT (LoRA)
- TRL (SFTTrainer)
- Datasets
- PyTorch

---

# 📂 Project Structure

```
Vedaz-Assignment/
│
├── configs/
│   ├── training_config.py
│  
│
├── dataset/
│   └── train.jsonl
│
├── docs/
│   └── VPS_vLLM_Hosting.md
│
├── outputs/
│   ├── adapter_model.safetensors
│   ├── adapter_config.json
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   ├── special_tokens_map.json
│   ├── chat_template.jinja
│   └── README.md
│
├── src/
│   ├── train.py
│   ├── inference.py
│  
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 📊 Dataset

The training dataset consists of structured multi-turn conversations in **JSONL** format.

Each sample follows the format:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "System prompt..."
    },
    {
      "role": "user",
      "content": "User query"
    },
    {
      "role": "assistant",
      "content": "Assistant response"
    }
  ]
}
```

The dataset contains:

- Empathetic conversations
- Kundli analysis
- Career guidance
- Marriage predictions
- Relationship guidance
- Financial queries
- Safety-aware responses
- Manually created conversations

---

# 🚀 Fine-Tuning Approach

The model was fine-tuned using **LoRA (Low-Rank Adaptation)**.

### Configuration

| Parameter | Value |
|-----------|-------|
| Method | LoRA |
| Epochs | 3 |
| Batch Size | 1 |
| Gradient Accumulation | 4 |
| Learning Rate | 2e-4 |

Only the LoRA adapter weights were trained while keeping the base model frozen, making the training memory-efficient.

---

# 📈 Training

Run:

```bash
python src/train.py
```

The script performs:

- Dataset loading
- Tokenization
- Chat template formatting
- LoRA adapter initialization
- Supervised Fine-Tuning
- Model saving

---

# 💬 Inference

Run:

```bash
python src/inference.py
```

The inference script loads:

- Base model
- LoRA adapter
- Tokenizer

and generates astrology responses for user queries.

---

# 📁 Model Output

The trained adapter is stored inside:

```
outputs/
```

Generated files include:

- adapter_model.safetensors
- adapter_config.json
- tokenizer.json
- tokenizer_config.json
- special_tokens_map.json
- chat_template.jinja

---

# 📄 Documentation

The repository also includes:

- VPS Hosting using vLLM
- Model deployment steps
- Manual training conversations

located in

```
docs/
```

---

# 📸 Training Result

The model completed fine-tuning successfully.

Example output:

```
Training completed!

Model saved successfully at:

outputs/
```


