from scipy.io import arff
import pandas as pd
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)



def load_dataset(filename):
    with open(filename, 'r') as f:
        data, meta = arff.loadarff(f)
    data = pd.DataFrame(data)
    X = data.drop(columns=['id', 'outlier'])
    # Map dataframe to encode values and put values into a numpy array
    y = data["outlier"].map(lambda x: 1 if x == b'yes' else 0).values
    return X, y

