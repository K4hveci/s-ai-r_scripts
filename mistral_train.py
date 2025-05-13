import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

# 1. CSV dosyasını oku
df = pd.read_csv("PoetryFoundationData.csv")

# 2. Prompt-output şeklinde dönüştür
def build_prompt(row):
    return {
        "text": f"""### User:
Write a poem titled "{row['Title']}" by {row['Poet']} tagged with {row['Tags']}.

### Assistant:
{row['Poem']}"""
    }

# 3. DataFrame'i prompt formatına çevir
data = df.apply(build_prompt, axis=1).tolist()
dataset = Dataset.from_list(data)

# 4. Model ve tokenizer'ı yükle
model_name = "RedHatAI/Mistral-7B-Instruct-v0.3-quantized.w4a16"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, device_map="auto")

tokenizer.pad_token = tokenizer.eos_token
# 5. LoRA ayarları
model = prepare_model_for_kbit_training(model)
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],  # bazı modellerde "k_proj", "o_proj" de eklenebilir
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_config)

# 6. Tokenizasyon
def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)

tokenized_dataset = dataset.map(tokenize, batched=True)

# 7. Eğitim ayarları
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    num_train_epochs=1,
    save_strategy="epoch",
    logging_steps=10,
    fp16=True,
    save_total_limit=1,
    logging_dir="./logs",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

# 8. Eğitimi başlat
trainer.train()
