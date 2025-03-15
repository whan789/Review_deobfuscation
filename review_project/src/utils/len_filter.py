#%%
import pandas as pd
from src.utils import data_load

def filter(result_path, test_path):

    # 첫번째 추론 결과 file_path
    result_df = data_load.data_load(result_path)
    # test set file_path
    test_df = data_load.data_load(test_path)

    test_df['input_length'] = test_df['input'].apply(lambda x: len(x))
    result_df['output_length'] = result_df['output'].apply(lambda x: len(x))

    # ID 기준으로 병합
    merged_df = test_df.merge(result_df[['ID', 'output_length']], on='ID', how='left')

    # 길이가 다른 행만 필터링 (test_df에서 유지)
    filtered_test = merged_df[merged_df['input_length'] != merged_df['output_length']].copy()

    # 필요 없는 열 정리
    filtered_test = filtered_test.drop(columns=['output_length', 'input_length'])

    return filtered_test

    

