from src.utils import data_load

def merging(infer_path, reinfer_df):
    infer_df = data_load.data_load(infer_path)
    updated_df = infer_df.set_index("ID").combine_first(reinfer_df.set_index("ID")).reset_index().drop('input',axis=1)

    return updated_df

