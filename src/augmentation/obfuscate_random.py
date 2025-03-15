import pandas as pd
import random
import re
from tqdm import tqdm

combined_df = pd.read_csv('/home/yooyoung/dacon/replace_insert_combined.csv',encoding='utf-8-sig')
train = pd.read_csv('/home/yooyoung/dacon/train.csv', encoding='utf-8-sig')
# 난독화 함수 정의
def obfuscate_sentence(sentence, train):
    words = sentence.split()  # 문장 내 단어 분리
    word_to_obfuscation = {}

    for word in words:
        # output 열에서 해당 단어를 포함한 행 필터링
        matching_rows = train[train['output'].str.strip().str.contains(re.escape(word), na=False)]

        if not matching_rows.empty:
            matching_input_forms = []
            for _, row in matching_rows.iterrows():
                input_words = row['input'].split()
                output_words = row['output'].split()
                # output 단어와 동일한 위치의 input 단어 선택
                for input_word, output_word in zip(input_words, output_words):
                    if output_word == word:
                        matching_input_forms.append(input_word)

            if matching_input_forms:
                # 난독화된 형태를 랜덤하게 선택
                random_obfuscation = random.choice(matching_input_forms)
                word_to_obfuscation[word] = random_obfuscation
            else:
                word_to_obfuscation[word] = word
        else:
            word_to_obfuscation[word] = word

    # 새로운 난독화된 문장 생성
    obfuscated_sentence = " ".join([word_to_obfuscation[word] for word in words])
    return obfuscated_sentence

# tqdm을 활용하여 combined_df의 sentence 컬럼을 난독화
combined_df['obfuscated_sentence'] = [
    obfuscate_sentence(sentence, train) for sentence in tqdm(combined_df['sentence'], desc="Obfuscating Sentences")
]

# 결과를 CSV 파일로 저장
output_csv_path = "/home/yooyoung/dacon/obfuscated_insert_replace_random.csv"
combined_df.to_csv(output_csv_path, encoding='utf-8-sig', index=False)

# 저장 완료 메시지
print(f"난독화된 데이터가 저장되었습니다: {output_csv_path}")
