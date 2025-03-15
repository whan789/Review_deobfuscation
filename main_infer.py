import pandas as pd
from review_project.src.utils import inference
from src.utils import data_load


if __name__== '__main__': 
    # 데이터 로드
    test_path = '/data/review/review_project/data/test.csv'
    test = data_load.data_load(test_path)

    FINETUNE_MODEL = "/data/review/review_project/src/models/dacon_llama_8b"
    BASE_MODEL = "MLP-KTLim/llama-3-Korean-Bllossom-8B"
    # FINETUNE_MODEL = "/data/review/review_project/src/models/dacon_llama_3b"
    # BASE_MODEL = "/data/review/review_project/src/models/dacon_llama_3b_base"

    print("Running inference...")
    restored_reviews = inference.inference(FINETUNE_MODEL, BASE_MODEL, test)

    # 결과 저장
    sample = pd.read_csv('/home/jinmin/data/sample_submission.csv', encoding = 'utf-8-sig')
    sample['output'] = restored_reviews