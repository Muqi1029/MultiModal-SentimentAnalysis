import json
import pandas as pd
from constant import TEST_LABEL
from sklearn.model_selection import train_test_split


def generate_json(path, val_rate, seed):
    data = pd.read_csv(path, sep=',')
    data['label'] = data['tag'].apply(lambda n: TEST_LABEL[n])
    x_train, x_val, y_train, y_val = train_test_split(data['guid'].values, data['label'].values,
                                                      test_size=val_rate, random_state=seed)
    train, test = [], []
    for guid, label in zip(x_train, y_train):
        train.append({
            'guid': int(guid),
            'label': int(label)
        })
    for guid, label in zip(x_val, y_val):
        test.append({
            'guid': int(guid),
            'label': int(label)
        })
    json.dump(train, open('train.json', 'w+', encoding='utf-8'))
    json.dump(test, open('val.json', 'w+', encoding="utf-8"))
    print("Finish")


if __name__ == '__main__':
    generate_json("../train.txt", 0.2, 1021)
