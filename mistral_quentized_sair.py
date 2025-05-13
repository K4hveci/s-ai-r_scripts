import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import gradio as gr

model_path = "./results/checkpoint-3463"

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# bitsandbytes quantization config (eğer modelin 4-bit/8-bit ise)


# Modeli GPU'ya yükle
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    trust_remote_code=True
)

# Metin üretme
def generate_response(prompt):
    inputs = tokenizer("write a poem about ", prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Gradio arayüzü
iface = gr.Interface(
    fn=generate_response,
    inputs=gr.Textbox(label="Prompt", placeholder="Enter what kind of poem do you want: "),
    outputs=gr.Textbox(label="Modelin Cevabı"),
    title="Quantized Mistral Model Testi",
    description="Quantized Mistral-7B modelini test etmek için metin girin."
)

iface.launch()
