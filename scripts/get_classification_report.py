from check_result_model_without_logs import detect_class
from sklearn.metrics import classification_report
import pandas as pd
from tqdm import tqdm


def predict(df: pd.DataFrame, path_to_weights: str) -> pd.DataFrame:
    predict_list = []
    for video in tqdm(df['path_to_video']):
        result = detect_class(video, path_to_weights)
        predict_list.append(result)
    df['predict'] = predict_list
    return df


def print_classfication_report(df: pd.DataFrame, name: str):
    print(f'{name}')
    print(classification_report(y_true=df['true_label'], y_pred=df['predict']))


if __name__ == '__main__':
    print("Введите 'train' или 'test'")
    sample = input()
    weights_path = '../weights/weights_ver5.pt'

    bridge_down = pd.read_csv(f'../dataframes/bridge_down_{sample}.csv')
    bridge_up = pd.read_csv(f'../dataframes/bridge_up_{sample}.csv')
    no_action = pd.read_csv(f'../dataframes/no_action_{sample}.csv')
    train_in_out = pd.read_csv(f'../dataframes/train_out_{sample}.csv')

    columns = ['video_name', 'true_label', 'path_to_video']

    all_data = pd.concat((bridge_down[columns],
                          bridge_up[columns],
                          no_action[columns],
                          train_in_out[columns]),
                         axis=0, ignore_index=True)

    result = predict(df=all_data, path_to_weights=weights_path)
    print_classfication_report(df=result, name=sample)
