import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel

def model(model_path, base_model):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Fine-tuned model not found at {model_path}")
    
    # Set specific device mapping configuration
    device_map = {"": 0}  # Map all modules to GPU 0, or customize as needed
    
    finetune_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device_map,  # Use explicit mapping
        torch_dtype=torch.float16
    )
    
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    text_gen_pipeline = pipeline(
        "text-generation",
        model=finetune_model,
        tokenizer=tokenizer
    )
    
    return tokenizer, text_gen_pipeline

def model_reinfer(model_path, base_model):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Fine-tuned model not found at {model_path}")
    
    device_map = {"": 0}  # Map all modules to GPU 0, or customize as needed

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map=device_map,
        torch_dtype=torch.float16
    )

    finetune_model = PeftModel.from_pretrained(base_model, model_path)

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # 텍스트 생성 파이프라인 (이제 정상 동작)
    text_gen_pipeline = pipeline(
        "text-generation",
        model=finetune_model,
        tokenizer=tokenizer,
        device_map="auto"
    )

    return tokenizer, text_gen_pipeline


    