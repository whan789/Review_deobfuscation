import pandas as pd
import torch 
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from datasets import Dataset
from transformers import TrainingArguments
import os
from tqdm.auto import tqdm
from transformers import pipeline
import numpy as np
from peft import PeftModel
import re

# CUDA 디바이스 설정
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # GPU ID를 설정합니다

# 모델 경로 확인
FINETUNE_MODEL = "/data/review/review_project/src/models/dacon_llama_8b" # 통합 모델
if not os.path.exists(FINETUNE_MODEL):
    raise FileNotFoundError(f"Fine-tuned model not found at {FINETUNE_MODEL}")

BASE_MODEL = "MLP-KTLim/llama-3-Korean-Bllossom-8B"

# 모델 및 토크나이저 로드 
finetune_model = AutoModelForCausalLM.from_pretrained(
    FINETUNE_MODEL,
    device_map="auto",
    torch_dtype=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL,trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# 텍스트 생성 파이프라인 
text_gen_pipeline = pipeline(
    "text-generation",
    model=finetune_model,
    tokenizer=tokenizer,
    # device_map=3
)

# 후처리 함수 
def postprocess_output(input_text, output_text):
    """입력과 동일한 공백을 유지하면서 단어 길이를 맞추는 후처리 함수"""
    output_text = re.sub(r'(<end_of_turn>|<start_of_turn>Assistant:|Output:)', '', output_text).strip()
    input_parts = re.split(r'(\s+)', input_text)  # 공백도 유지
    output_parts = re.split(r'(\s+)', output_text)  # 공백도 유지

    min_length = min(len(input_parts), len(output_parts))
    input_parts = input_parts[:min_length]
    output_parts = output_parts[:min_length]

    corrected_output = []
    for i in range(min_length):
        if input_parts[i].strip():  # 단어인 경우
            min_word_length = min(len(input_parts[i]), len(output_parts[i]))
            corrected_output.append(output_parts[i][:min_word_length])
        else:  # 공백인 경우 그대로 유지
            corrected_output.append(input_parts[i])

    return ''.join(corrected_output)

# 문장 분할 함수
def split_sentence(text, max_length):
    """문장을 최대 길이 기준으로 분할"""
    sentences = re.findall(r'[^.!?]+[.!?]?', text)
    result, current_sentence = [], ""

    for part in sentences:
        part = part.strip()
        if len(current_sentence) + len(part) <= max_length:
            current_sentence += " " + part if current_sentence else part
        else:
            result.append(current_sentence.strip())
            current_sentence = part

    if current_sentence:
        result.append(current_sentence.strip())

    return result

# 리뷰 복원 함수 
def restore_reviews_sequential(test_df, text_gen_pipeline, tokenizer=None):
    """
    리뷰 복원 함수
    :param test_df: 데이터프레임 (input 열 포함)
    :param text_gen_pipeline: 텍스트 생성 파이프라인
    :param tokenizer: (선택) 토크나이저 (None일 경우 기본값 사용)
    :return: 복원된 리뷰 리스트
    """
    restored_reviews = []

    for index, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Processing Reviews", leave=False):
        query = row['input']
        split_queries = split_sentence(query, max_length=300)  # 50자 단위로 분할
        restored_parts = []

        for part in split_queries:
            prompt = (
                "<start_of_turn> Your task is to transform the given obfuscated Korean review into a clear, natural-sounding Korean review that reflects its original meaning.\n"
                "### Rules:\n"
                "1. The restored output must have the same number of characters and spacing as the input.\n"
                "2. Only correct unnatural expressions, typos, or grammar errors without changing word order.\n"
                "3. Avoid unnecessary repetitions and preserve sentence fluency.\n"
                f"Input: {part}\n"
                "<end_of_turn>\n"
                "<start_of_turn>Assistant:\n"
                "Output:"
            )

            try:
                generated = text_gen_pipeline(
                    prompt,
                    num_return_sequences=1,
                    num_beams=5,
                    max_new_tokens=len(part) + 5,
                    do_sample=False,
                    eos_token_id=tokenizer.eos_token_id if tokenizer else None,
                    pad_token_id=tokenizer.pad_token_id if tokenizer else None
                )

                # `generated`의 구조 확인 후 `generated_text` 키 확인
                if generated and 'generated_text' in generated[0]:
                    generated_text = generated[0]['generated_text']
                else:
                    generated_text = ""

                # "Output:" 이후의 텍스트 추출 (없을 경우 대비)
                output_start = generated_text.find("Output:")
                if output_start != -1:
                    output = generated_text[output_start + len("Output:"):].strip()
                else:
                    output = generated_text.strip()  # "Output:"이 없으면 전체 문장 사용

                # 후처리 적용
                output = postprocess_output(part, output)
                restored_parts.append(output)

            except Exception as e:
                print(f"Error processing text at index {index}: {e}")
                restored_parts.append(part)  # 오류 발생 시 원본 유지

        final_output = " ".join(restored_parts)
        restored_reviews.append(final_output)

    return restored_reviews

# 데이터 로드
test_file = '/home/jinmin/data/test.csv'
if not os.path.exists(test_file):
    raise FileNotFoundError(f"Test CSV file not found at {test_file}")
test = pd.read_csv(test_file, encoding='utf-8-sig')

# 리뷰 복원 실행
restored_reviews = restore_reviews_sequential(
    test_df=test,
    text_gen_pipeline=text_gen_pipeline,
    tokenizer=tokenizer
)

# 결과 저장
sample = pd.read_csv('/home/jinmin/data/sample_submission.csv', encoding = 'utf-8-sig')
test_list = restored_reviews
sample['output'] = test_list
sample.to_csv('/home/jinmin/results/llama_long9.csv', index=False, encoding='utf-8-sig')
