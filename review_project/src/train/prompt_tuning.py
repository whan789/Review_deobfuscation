import pandas as pd
import torch 
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments, EarlyStoppingCallback
from trl import SFTTrainer
from peft import PeftModel
import optuna
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model, PeftModel
from datasets import Dataset
from transformers import EarlyStoppingCallback
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator, get_linear_schedule_with_warmup
from peft import get_peft_config, get_peft_model, PromptTuningInit, PromptTuningConfig, TaskType, PeftType
from datasets import load_dataset
import os
from torch.utils.data import DataLoader
from tqdm import tqdm


# CUDA 디바이스 설정
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # GPU ID를 설정합니다

train = pd.read_excel('/home/alfee050523/obfucation_gongmo/data/train_merged2.xlsx', dtype=str, engine='openpyxl')
train = train.drop(columns=["id"])  # id 컬럼 제거
dataset = Dataset.from_pandas(train)

train_test_split = dataset.train_test_split(test_size=0.2, seed=42)

train_dataset = train_test_split['train']
test_dataset = train_test_split['test']

# 모델 로드
model_name = "/home/alfee050523/obfucation_gongmo/llama-3.2-Korean-Bllossom-3B_768_best"

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

peft_config = PromptTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    prompt_tuning_init=PromptTuningInit.TEXT,
    num_virtual_tokens=512,
    prompt_tuning_init_text="Your task is to transform the given obfuscated Korean review into a clear, correct, and natural-sounding Korean review that reflects its original meaning.:",
    tokenizer_name_or_path="/home/alfee050523/obfucation_gongmo/llama-3.2-Korean-Bllossom-3B_768_best",
)

model = get_peft_model(model, peft_config)

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
    peft_config=peft_config,
    formatting_func=lambda x: x['input_ids'],
    callbacks=[early_stopping]  # Early Stopping 추가
)

#  파인튜닝 시작
trainer.train()

ADAPTER_MODEL = "/home/review/ksw/model_file/llama-3.2-Korean-Bllossom-3B_best_hyperparams_with_prompt_tuning"

trainer.model.save_pretrained(ADAPTER_MODEL)

BASE_MODEL = "/home/review/model_file/llama-3.2-Korean-Bllossom-3B_768_best"

model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map='auto', torch_dtype=torch.float16)
model = PeftModel.from_pretrained(model, ADAPTER_MODEL, device_map='auto', torch_dtype=torch.float16)

# BASE_MODEL과 ADAPTER_MODEL이 통합된 상태로 저장
model.save_pretrained("/home/review/ksw/model_file/llama-3.2-Korean-Bllossom-3B_best_hyperparams_with_prompt_tuning_final")
tokenizer.save_pretrained("/home/review/ksw/model_file/llama-3.2-Korean-Bllossom-3B_best_hyperparams_with_prompt_tuning_final")