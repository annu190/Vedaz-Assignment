import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ==========================
# PATHS
# ==========================

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_PATH = "outputs"

# ==========================
# LOAD TOKENIZER
# ==========================

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True
)

# ==========================
# LOAD BASE MODEL
# ==========================

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

# ==========================
# LOAD LORA ADAPTER
# ==========================

model = PeftModel.from_pretrained(model, ADAPTER_PATH)

model.eval()

# ==========================
# CHAT FUNCTION
# ==========================

def chat(user_input):
    messages = [
        {
            "role": "system",
            "content": "You are Vedaz's AI Vedic astrologer. Give compassionate guidance."
        },
        {
            "role": "user",
            "content": user_input
        }
    ]

    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.9
        )

    response = tokenizer.decode(output[0], skip_special_tokens=True)

    return response

# ==========================
# RUN LOOP
# ==========================

print("\nVedaz AI Astrologer Chatbot Ready!\n(Type 'exit' to stop)\n")

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        break

    response = chat(user_input)
    print("\nAstrologer:", response, "\n")