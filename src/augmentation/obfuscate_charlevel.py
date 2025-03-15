import pandas as pd
import random
import re
from tqdm import tqdm
from collections import defaultdict

# 데이터 로드
combined_df = pd.read_csv('/home/yooyoung/dacon/df_insert_cleaned.csv', encoding='utf-8-sig')
train = pd.read_csv('/home/yooyoung/dacon/train.csv', encoding='utf-8-sig')

def obfuscate_sentence_char(sentence, train):
    """
    주어진 문장을 train 데이터의 output과 input을 기반으로 **문자 단위**로 난독화하며,
    각 문자를 변환할 때 **랜덤한 난독화 형태를 적용**합니다.

    Args:
        sentence (str): 원본 문장
        train (DataFrame): input과 output 열이 포함된 데이터프레임

    Returns:
        str: 난독화된 문장
    """
    char_to_obfuscation = {}

    for char in sentence:
        if char.strip():  # 공백은 변환하지 않음
            # output 열에서 해당 문자를 포함한 행 필터링
            matching_rows = train[train['output'].str.contains(re.escape(char), na=False, regex=True)]

            if not matching_rows.empty:
                matching_input_chars = []
                for _, row in matching_rows.iterrows():
                    input_text = row['input']
                    output_text = row['output']
                    
                    min_len = min(len(input_text), len(output_text))
                    for i in range(min_len):
                        if output_text[i] == char:
                            matching_input_chars.append(input_text[i])

                if matching_input_chars:
                    # 랜덤하게 난독화된 문자 선택
                    random_obfuscation = random.choice(matching_input_chars)
                    char_to_obfuscation[char] = random_obfuscation
                else:
                    char_to_obfuscation[char] = char
            else:
                char_to_obfuscation[char] = char
        else:
            char_to_obfuscation[char] = char  # 공백 유지

    # 새로운 난독화된 문장 생성
    obfuscated_sentence = "".join([char_to_obfuscation[char] for char in sentence])
    return obfuscated_sentence

# tqdm을 활용하여 문장 난독화 진행
tqdm.pandas()
combined_df['obfuscated_sentence'] = combined_df['sentence'].progress_apply(lambda x: obfuscate_sentence_char(x, train))

# 결과를 CSV 파일로 저장
output_csv_path = "/home/yooyoung/dacon/obfuscated_insert_random_char.csv"
combined_df.to_csv(output_csv_path, encoding='utf-8-sig', index=False)

print(f"난독화된 데이터가 저장되었습니다: {output_csv_path}")
