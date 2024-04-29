from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from itertools import combinations_with_replacement


def divide_data(data, train_size=0.6, val_size=0.2):
    test_size = 1 - train_size - val_size
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=42)
    train_data, val_data = train_test_split(train_data, test_size=val_size / (train_size + val_size), random_state=42)
    return train_data, val_data, test_data


def add_normalized_features(data, remove_original=False):
    res_data = data.copy()
    features = data.drop(columns=['y', 'const']).columns

    normalized_features = (data[features] - data[features].mean()) / data[features].std()

    res_data = pd.concat([res_data, normalized_features.add_suffix('_norm')], axis=1)

    print(f"added {len(features)} features")

    if remove_original:
        res_data.drop(columns=features, inplace=True)
    return res_data


def add_polynomial_features(data, degree):
    res_data = data.copy()
    features = data.drop(columns=['y', 'const']).columns

    new_features = {}
    while degree > 1:
        to_add = list(combinations_with_replacement(features, r=degree))

        for comb in to_add:
            new_feature_name = '_'.join(comb)
            new_feature = 1
            for feature in comb:
                new_feature *= data[feature]
            new_features[new_feature_name] = new_feature

        degree -= 1

    # Concatenate original data and new features
    res_data = pd.concat([res_data] + [pd.Series(data=new_features[feature], name=feature) for feature in new_features],
                         axis=1)

    print(f"added {len(to_add)} features")
    return res_data


def add_transform_features(data, transforms):
    res_data = data.copy()
    features = data.drop(columns=['y', 'const']).columns
    new_columns = []
    new_data = []
    for transform, name in transforms:
        for feature in features:
            new_columns.append(f"{name}_{feature}")
            transformed_feature = pd.Series(transform(data[feature]), name=f"{name}_{feature}")
            new_data.append(transformed_feature)
    res_data = pd.concat([res_data] + new_data, axis=1)
    print(f"added {len(features) * len(transforms)} features")
    return res_data


def divide_y(data):
    return data.drop(columns=['y']).values, data['y'].values

