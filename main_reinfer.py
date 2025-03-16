import pandas as pd
from src.utils import inference, len_filter, merging


if __name__== '__main__': 
    # test set 경로
    test_path = 'test_path'
    # 첫번째 추론 결과 파일 경로
    result_path = 'inference_path'


    FINETUNE_MODEL = "finetuned_model"
    BASE_MODEL = "base_model"
    
    # test input과 길이가 다른 추론 결과 행 필터링
    filtered_test = len_filter.filter(test_path,result_path)

    # 필터링된 행 재추론
    restored_reviews = inference.inference_kiwi(FINETUNE_MODEL, BASE_MODEL, filtered_test)
    print("Running reinference...")

    # 재추론 결과 파일과 첫번째 추론 결과 파일 병합
    merged_df = merging.merging(result_path, restored_reviews)

    # 결과 저장
    sample = pd.read_csv('sample_path', encoding = 'utf-8-sig')
    sample['output'] = merged_df['output']