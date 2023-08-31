import pandas as pd

def load_data(data_dir='../data/'):

    data = pd.read_csv(data_dir + 'winequality-white.csv', sep=';')
    y = data['quality'].values
    X = data.drop(['quality'], axis=1).values

    print("Data:", X.shape)

    return X, y