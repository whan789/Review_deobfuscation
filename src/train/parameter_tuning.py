import optuna
import torch
import os
import json
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model, PeftModel
from trl import SFTTrainer
from datasets import Dataset
from transformers import EarlyStoppingCallback
import gc

# CUDA 디바이스 설정
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # GPU ID를 설정
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["FLASH_ATTENTION"] = "1"

# 데이터 로드
train = pd.read_csv('train_path', encoding='utf-8-sig')
train = train.drop(columns=['Unnamed: 0', 'ID'])  # id 컬럼 제거
dataset = Dataset.from_pandas(train)

train_test_split = dataset.train_test_split(test_size=0.2, seed=42)

train_dataset = train_test_split['train']
test_dataset = train_test_split['test']

# 모델 로드
model_name = "base_model"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Tokenizer 설정
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'right'

# Prompt 생성 함수
def create_prompt(input, output):
    prompt = (
        "<start_of_turn> Your task is to transform the given obfuscated Korean review into a clear, correct, and natural-sounding Korean review that reflects its original meaning.\n"
        f"Input: {input}\n"
        "<end_of_turn>\n"
        "<start_of_turn>Assistant:\n"
        f"Output: {output}"
    )
    return prompt

# 데이터셋 포맷팅
def format_chat_template(row):
    prompt = create_prompt(row["input"], row["output"])
    tokens = tokenizer.encode(prompt, truncation=True, max_length=768)
    row["input_ids"] = tokens
    return row

train_dataset = train_dataset.map(format_chat_template, batched=False, num_proc=1)
test_dataset = test_dataset.map(format_chat_template, batched=False, num_proc=1)

# 최적 하이퍼파라미터 저장 파일 경로
BEST_PARAMS_FILE = "param_path"

def objective(trial):
    global BEST_PARAMS_FILE  # 최적 파라미터 저장 경로

    # 학습 시작 전 메모리 정리
    torch.cuda.empty_cache()
    gc.collect()

    # 튜닝할 하이퍼파라미터 정의
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 5e-4)
    lora_r = trial.suggest_categorical("lora_r", [8, 16]) 
    lora_alpha = trial.suggest_categorical("lora_alpha", [16, 32, 64])
    lora_dropout = trial.suggest_uniform("lora_dropout", 0.05, 0.3)

    # 모델 로드 (float16 적용)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)

    # LoRA 설정
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
        lora_dropout=lora_dropout,
        bias='none',
        task_type='CAUSAL_LM'
    )

    # LoRA 적용
    model = get_peft_model(model, lora_config)
    model.train()

    # TrainingArguments 설정
    training_args = TrainingArguments(
        output_dir="./results_tuning",
        num_train_epochs=3,  
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,  # 기존 2 → 4 (메모리 절약)
        optim="adamw_bnb_8bit",
        learning_rate=learning_rate,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=1,
        logging_dir="./logs",
        fp16=True,
        load_best_model_at_end=True,
    )

    # Early Stopping 추가
    early_stopping = EarlyStoppingCallback(early_stopping_patience=2)

    # Trainer 설정
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        args=training_args,
        peft_config=lora_config,
        callbacks=[early_stopping]
    )

    # 학습 시작
    trainer.train()

    # 평가 (eval_loss 반환)
    eval_loss = trainer.evaluate()["eval_loss"]

    # 최적 하이퍼파라미터 저장 (손실 값 포함)
    best_params = trial.params
    best_params["eval_loss"] = eval_loss  # 손실 값 추가

    with open(BEST_PARAMS_FILE, "w") as f:
        json.dump(best_params, f, indent=4)

    print(f"🔹 현재 최적 파라미터: {best_params}")

    # 메모리 정리 (모델 삭제)
    del model
    torch.cuda.empty_cache()
    gc.collect()

    return eval_loss  # Optuna가 최소화할 값 (낮을수록 좋음)

# Optuna 최적화 실행 (Pruner 적용)
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=5)  # 5번 시도

# 최적 하이퍼파라미터 출력
print("Best Hyperparameters:", study.best_params)

# 중간 저장된 최적 하이퍼파라미터 로드
with open(BEST_PARAMS_FILE, "r") as f:
    best_params = json.load(f)

# # 최적 하이퍼파라미터 저장
# best_params = study.best_params

# 최적 조합으로 모델 재학습
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

lora_config = LoraConfig(
    r=best_params["lora_r"],
    lora_alpha=best_params["lora_alpha"],
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
    lora_dropout=best_params["lora_dropout"],
    bias='none',
    task_type='CAUSAL_LM'
)

model = get_peft_model(model, lora_config)
model.train()

training_args = TrainingArguments(
    output_dir="./results_best",
    num_train_epochs=10,  # 최종 학습에서는 10 Epoch
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=best_params["learning_rate"],
    eval_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=1000,
    save_total_limit=2,  # 저장할 체크포인트 수 제한
    logging_dir="./logs",
    fp16=True,
    load_best_model_at_end=True,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    args=training_args,
    peft_config=lora_config
)

# 최적 하이퍼파라미터 기반 최종 학습 시작
trainer.train()

# 최종 모델 저장
ADAPTER_MODEL = "adapter_path"
trainer.model.save_pretrained(ADAPTER_MODEL)

BASE_MODEL = "MLP-KTLim/llama-3-Korean-Bllossom-8B"
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map='auto', torch_dtype=torch.float16)
model = PeftModel.from_pretrained(model, ADAPTER_MODEL, device_map='auto', torch_dtype=torch.float16)

# 최적 모델 저장
FINAL_MODEL_PATH = "best_model"
model.save_pretrained(FINAL_MODEL_PATH)
tokenizer.save_pretrained(FINAL_MODEL_PATH)