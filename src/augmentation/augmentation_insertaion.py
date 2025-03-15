import pandas as pd
from utils.BertAugmentation import BertAugmentation  # BERT 모델을 활용한 데이터 증강 유틸리티 (사용자 정의 모듈)
from tqdm import tqdm
import joblib  # 병렬 처리를 위한 라이브러리
from functools import partial  # 함수의 일부 매개변수를 고정시켜 새로운 함수를 생성

tqdm.pandas()

BERT_aug = BertAugmentation()  # BERT 증강 객체 생성

def apply_augmentation(df, aug_func, n_jobs=8):
    # df: 증강할 데이터
    # aug_func: 적용할 증강 함수(예: random_masking_insertion)
    # n_jobs: 병렬 작업 수
    """augmentation 병렬 처리"""
    pool = joblib.Parallel(n_jobs=n_jobs, prefer='threads')
    mapper = joblib.delayed(aug_func)
    
    tasks_1 = [mapper(row) for row in df['output']]
    # tasks_2 = [mapper(row) for row in df['sentence2']]
    
    df['output'] = pool(tqdm(tasks_1))
    # df['sentence2'] = pool(tqdm(tasks_2))
    
    return df

def save_augmented_dataset(df, filename):
    """중복 제거 후 증강된 데이터셋 저장"""
    # df.drop_duplicates(['sentence1', 'sentence2'], inplace=True)
    df.drop_duplicates(['output'], inplace=True)
    df.reset_index(drop=True).to_json(filename)

# Random masking insertion
# 주어진 문장(sentence)에 대해 15% 비율로 랜덤 마스킹 삽입을 적용
def random_masking_insertion(sentence, ratio=0.15):
    return BERT_aug.random_masking_insertion(sentence, ratio=ratio)

if __name__ == "__main__":
    # orig_train = pd.read_json('sts/datasets/klue-sts-v1.1_train.json')  # KLUE-STS 학습 데이터셋(JSON 형식) 로드
    orig_train = pd.read_csv('/home/jinmin/data/train_filtered.csv')  # train 데이터셋 로드
    # Apply random masking insertion
    # 원본 데이터를 복사한 후, random_masking_insertion 함수를 적용한 뒤 증강
    # random_masking_insertion_train = orig_train.copy()
    # random_masking_insertion_train = orig_train[['sentence1']].copy()
    random_masking_insertion_train = orig_train[['output']].copy()
    # random_masking_insertion_train = orig_train[['output']].copy().head(100)

    # print(random_masking_insertion_train.columns)  # Index(['output'], dtype='object')
    # print(random_masking_insertion_train['output'][0:1])

    print(f"증강 전 데이터 수: {len(random_masking_insertion_train)}")
    random_masking_insertion_train = apply_augmentation(random_masking_insertion_train, partial(random_masking_insertion, ratio=0.15))
    print(f"증강 후 데이터 수: {len(random_masking_insertion_train)}")  # ?

    # 증강 데이터 저장
    # save_augmented_dataset(pd.concat([orig_train, random_masking_insertion_train]), 'sts/datasets/klue-sts-v1.1_train_random_masking_insertion_augset2.json')
    # save_augmented_dataset(pd.concat([orig_train[['sentence1']], random_masking_insertion_train]), 'sts/datasets/klue-sts-v1.1_train_random_masking_insertion_augset_sentence1.json')
    save_augmented_dataset(pd.concat([orig_train[['output']], random_masking_insertion_train]), 'sts/datasets/total_train_output_augmented.json')