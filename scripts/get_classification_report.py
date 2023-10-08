from check_result_without_logs import detect_class
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report


def predict(path_to_df: str) -> pd.DataFrame:
    df = pd.read_csv(path_to_df)
    predict_list = []
    for video in tqdm(df['path_to_video']):
        result = detect_class(video)
        predict_list.append(result)
    df['predict'] = predict_list
    return df


def print_classfication_report(df: pd.DataFrame):
    print(classification_report(y_true=df['true_label'], y_pred=df['predict']))


if __name__ == '__main__':
    path = '../dataframes/all_train.csv'
    # path = '../dataframes/all_test.csv'
    result = predict(path_to_df=path)
    print_classfication_report(result)
