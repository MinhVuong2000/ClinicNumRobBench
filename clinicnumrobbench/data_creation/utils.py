import pandas as pd


RANDOM_STATE = 42

def format_df(df):
    return df.apply(lambda r: {
        "question": "\n\n".join([r['context'], r['question']]),
        "answer": r['answer'],
        # "open_ended_answer": r['answer'],
        "dataset_source": r['sub_type']
    }, axis=1).apply(pd.Series)
