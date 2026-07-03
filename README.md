# 🌟 Vedaz AI Astrologer Fine-Tuning Project

This project fine-tunes **Qwen2.5** using **LoRA (Parameter Efficient Fine Tuning)** to build a Vedic astrology chatbot.

## 🚀 Tech Stack

- Qwen2.5-0.5B-Instruct
- HuggingFace Transformers
- TRL (SFTTrainer)
- PEFT (LoRA)
- PyTorch
- Google Colab / VS Code

---

## 📁 Project Structure


dataset/
train.jsonl

src/
train.py

inference.py

outputs/
(fine-tuned model)

requirements.txt

README.md


---

## ⚙️ Training Steps

1. Install dependencies
pip install -r requirements.txt

2. Run training
python src/train.py

🧠 Model Details
Base Model: Qwen2.5-0.5B-Instruct

Fine-tuning: LoRA

Dataset: Custom Vedic Astrology conversations

Task: Instruction tuning chatbot

💬 Run Chatbot
python src/inference.py

☁️ Deployment (vLLM - VPS)
The model can be served using vLLM:

-pip install vllm

 python -m vllm.entrypoints.openai.api_server \
  --model outputs \
  --port 8000
  
📌 Features
Kundli-based conversational astrology
Empathetic responses
Non-fatalistic predictions
Safe AI behavior for emotional queries
⚠️ Disclaimer

This AI is for educational and entertainment purposes only.
It does not provide real astrological or medical advice.

# 3. VPS + vLLM WRITEUP 

# Hosting Fine-tuned Model using vLLM on VPS

## Step 1: Setup VPS
- Ubuntu 22.04 recommended
- Install Python, pip, CUDA drivers

## Step 2: Install dependencies
pip install vllm transformers accelerate
## Step 3: Upload model
Copy outputs/ folder to VPS using SCP or GitHub.
## Step 4: Start vLLM server
python -m vllm.entrypoints.openai.api_server \
  --model outputs \
  --host 0.0.0.0 \
  --port 8000
## Step 5: Access API
http://<your-vps-ip>:8000/v1/chat/completions
# Benefits of vLLM
-High-speed inference

-Low latency

-Efficient memory usage

-Production-ready LLM serving
