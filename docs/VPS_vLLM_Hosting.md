# Hosting the Fine-Tuned Qwen2.5 Model on a VPS using vLLM

## Objective

The objective is to deploy the fine-tuned Qwen2.5 model on a Virtual Private Server (VPS) using vLLM for efficient and scalable inference.

---

## Prerequisites

- Ubuntu 22.04 VPS
- NVIDIA GPU (recommended)
- Python 3.10+
- CUDA Toolkit
- Git
- Internet connection

---

## Step 1: Update the Server

```bash
sudo apt update
sudo apt upgrade -y
```

---

## Step 2: Install Python

```bash
sudo apt install python3 python3-pip git -y
```

Verify installation:

```bash
python3 --version
pip3 --version
```

---

## Step 3: Install Required Libraries

```bash
pip install torch transformers accelerate vllm
```

---

## Step 4: Upload the Fine-Tuned Model

Copy the trained model folder (`outputs/`) from the local machine to the VPS using SCP or Git.

Example:

```bash
scp -r outputs username@your_server_ip:/home/username/
```

---

## Step 5: Start the vLLM Server

Run:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model outputs \
    --host 0.0.0.0 \
    --port 8000
```

The server will start and expose an OpenAI-compatible REST API.

---

## Step 6: Test the API

Example request:

```bash
curl http://localhost:8000/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
"model":"outputs",
"messages":[
{"role":"user","content":"Mujhe career ke baare mein bataiye."}
]
}'
```

---

## Advantages of vLLM

- High-speed inference
- Efficient GPU memory utilization
- Supports large language models
- OpenAI-compatible API
- Easy deployment on cloud servers
- Suitable for production environments

---

## Conclusion

The fine-tuned Qwen2.5 model can be deployed efficiently on a VPS using vLLM. This setup provides fast response times, optimized GPU utilization, and an API that can be integrated with web or mobile applications.