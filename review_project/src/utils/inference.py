#%%
from src.utils import load_model, restore
from kiwipiepy import Kiwi
import pandas as pd
from tqdm import tqdm


def inference(model_path, base_model, test_df):
    tokenizer, text_gen_pipeline = load_model.model(model_path, base_model)

    restored_reviews = restore.restore_reviews_sequential(
        test_df = test_df,
        text_gen_pipeline=text_gen_pipeline,
        tokenizer=tokenizer
    )

    return restored_reviews

def inference_kiwi(model_path, base_model, test_df):
    kiwi = Kiwi()
    tokenizer, text_gen_pipeline = load_model.model_reinfer(model_path, base_model)

    # 결과를 저장할 리스트
    output_data = []

    # filtered_dict의 key는 id, value는 input
    for key, value in tqdm(test_df.items(), desc="Processing", unit="entry"):
        text = value
        sentences = kiwi.split_into_sents(text)
        inputs = [sent.text for sent in sentences]  # 문장 리스트 생성

        # 데이터프레임 생성
        df = pd.DataFrame(inputs, columns=['input'])

        # 리뷰 복원 실행
        restored_reviews = restore.restore_reviews_sequential(
            test_df=df,
            text_gen_pipeline=text_gen_pipeline,
            tokenizer=tokenizer
        )

        # 결과를 하나의 문장으로 합치기
        final_sentence = ' '.join(restored_reviews)

        # 결과 저장
        output_data.append({'id': key, 'output': final_sentence})
    return restored_reviews



