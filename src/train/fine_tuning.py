import pandas as pd
import torch 
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments, EarlyStoppingCallback
from trl import SFTTrainer
from peft import PeftModel
import os

# CUDA 디바이스 설정
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # GPU ID를 설정합니다

train = pd.read_csv('/home/review/data/combined_dataset.csv')

dataset = Dataset.from_pandas(train)

train_test_split = dataset.train_test_split(test_size=0.2, seed=42)

train_dataset = train_test_split['train']
test_dataset = train_test_split['test']

# 모델 로드
model_name = "Bllossom/llama-3.2-Korean-Bllossom-3B"

device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModelForCausalLM.from_pretrained(model_name).to(device) 
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'right'

def create_prompt(input, output):
    prompt = (
        "<start_of_turn> Your task is to transform the given obfuscated Korean review into a clear, correct, and natural-sounding Korean review that reflects its original meaning.\n"
        f"Input: {input}\n"
        "<end_of_turn>\n"
        "<start_of_turn>Assistant:\n"
        f"Output: {output}"
    )
    return prompt


def format_chat_template(row):
    prompt = create_prompt(row["input"], row["output"])
    tokens = tokenizer.encode(prompt, truncation=True, max_length=512)
    row["input_ids"] = tokens
    return row

# 데이터셋에 적용
train_dataset = train_dataset.map(format_chat_template, batched=False, num_proc=4)
test_dataset = test_dataset.map(format_chat_template, batched=False, num_proc=4)

lora_config = LoraConfig(
    r = 8,
    lora_alpha = 32,
    target_modules = [
    "q_proj", "v_proj", "k_proj", "o_proj", 
    "gate_proj", "down_proj", "up_proj"
],
    lora_dropout = 0.1,
    bias ='none',
    task_type ='CAUSAL_LM'
)

model = get_peft_model(model, lora_config)

# 모델을 훈련 모드로 설정
model.train()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# 훈련 인자 설정
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=10,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    optim="paged_adamw_32bit",
    eval_strategy="steps",
    eval_steps=500, # 모델의 평가 주기
    logging_dir="./logs",
    logging_steps=100,
    warmup_steps=10, # 학습률 스케줄링
    logging_strategy="steps",
    learning_rate=2e-4,
    group_by_length=True,
    fp16=True,
    load_best_model_at_end=True,  # Best Model 로드 활성화
    save_strategy="steps",
    save_steps=500,  # 모델 저장 주기
    save_total_limit=3  # 저장할 체크포인트 수 제한
)

# Early Stopping 콜백 설정
early_stopping = EarlyStoppingCallback(early_stopping_patience=3)
 
# Trainer 초기화 및 훈련
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    args=training_args,
    peft_config=lora_config,
    formatting_func=lambda x: x['input_ids'],
    callbacks=[early_stopping]  # Early Stopping 추가
)

#  파인튜닝 시작
trainer.train()

ADAPTER_MODEL = "./model_file/llama-3.2-Korean-Bllossom-3B_prompt_aug"

trainer.model.save_pretrained(ADAPTER_MODEL)

BASE_MODEL = "Bllossom/llama-3.2-Korean-Bllossom-3B"

model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map='auto', torch_dtype=torch.float16)
model = PeftModel.from_pretrained(model, ADAPTER_MODEL, device_map='auto', torch_dtype=torch.float16)

# BASE_MODEL과 ADAPTER_MODEL이 통합된 상태로 저장
model.save_pretrained("./model_file/llama-3.2-Korean-Bllossom-3B_prompt_finetuning_aug")
tokenizer.save_pretrained("./model_file/llama-3.2-Korean-Bllossom-3B_prompt_finetuning_aug")