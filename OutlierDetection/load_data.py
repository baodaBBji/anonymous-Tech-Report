import os
import numpy as np
from utils import *


def load_data(name, data_dir='../data/'):
    if not os.path.exists(data_dir):
        data_dir = '/data/wangyu/data/'
    if name == 'SpamBase':
        # Good
        # [0.13679283 0.09023855 0.05738381 0.05118552 0.04391536]
        filename = f'{data_dir}SpamBase_withoutdupl_norm_40.arff'
        X, y = load_dataset(filename=filename)
        print(np.shape(X), np.shape(y))
        K = 9
        N = 1679
        class_balance = [1 - N / 4207.0, N / 4207.0]
        lof_krange = list(range(10, 110, 10)) * 6
        knn_krange = list(range(10, 110, 10)) * 6
        if_range = [0.5, 0.6, 0.7, 0.8, 0.9] * 6
        # mahalanobis_N_range=[N]
        mahalanobis_N_range = [1400, 1500, 1600, 1700, 1800, 1900]
        if_N_range = np.sort(mahalanobis_N_range * 5)
        N_range = np.sort(mahalanobis_N_range * 10)

        print(N / len(y))


    elif name == 'PageBlock':
        # Not bad, not good
        # [0.53446205 0.27996802 0.1267319  0.03581897 0.01230324]
        filename = f'{data_dir}PageBlocks_norm_10.arff'
        X, y = load_dataset(filename=filename)
        print(np.shape(X), np.shape(y))
        K = 80
        N = 560
        # num_outliers = [N, N, N, N]
        class_balance = [0.9, 0.1]
        # lof_krange = [55, 60, 65, 70, 75]
        # knn_krange = [55, 60, 65, 70, 75]
        # if_range = [0.5, 0.6,0.7, 0.8,0.9]
        lof_krange = list(range(10, 110, 10)) * 6
        knn_krange = list(range(10, 110, 10)) * 6
        # lof_krange = range(70,90,4)
        # knn_krange = [60, 70, 80, 90, 100]
        if_range = [0.5, 0.6, 0.7, 0.8, 0.9] * 6
        # mahalanobis_N_range=[560]
        mahalanobis_N_range = [300, 400, 500, 600, 700, 800]
        # mahalanobis_N_range = [550, 560, 570, 580, 590, 600]
        N_size = 6
        if_N_range = np.sort(mahalanobis_N_range * 5)
        N_range = np.sort(mahalanobis_N_range * 10)

    elif name == 'Pima':
        # Unknown, too little points
        filename = f'{data_dir}Pima_withoutdupl_norm_35.arff'
        X, y = load_dataset(filename=filename)
        print(np.shape(X), np.shape(y))
        K = 100
        N = 268
        print(N / len(y))
        num_outliers = [N, N, N, N]
        class_balance = [1 - N / 768.0, N / 768.0]
        lof_krange = list(range(10, 210, 10)) * 6
        knn_krange = list(range(10, 210, 10)) * 6
        if_range = [0.5, 0.6, 0.7, 0.8, 0.9] * 6
        mahalanobis_N_range = [N]
        mahalanobis_N_range = [220, 230, 240, 250, 260, 270]

        if_N_range = np.sort(mahalanobis_N_range * 5)
        N_range = np.sort(mahalanobis_N_range * 20)

    elif name == 'Shuttle':
        # Good
        # [0.82128862 0.12354073 0.02420756 0.02181645 0.0064696 ]
        import hdf5storage
        mat = hdf5storage.loadmat(f'{data_dir}shuttle.mat')
        X = mat['X']
        y = mat['y']
        print(len(y))
        print(np.sum(y) / len(y))
        lof_krange = list(range(10, 110, 10)) * 6
        knn_krange = list(range(10, 110, 10)) * 6
        if_range = [0.5, 0.6, 0.7, 0.8, 0.9] * 6
        mahalanobis_N_range = [1000, 1500, 2000, 2500, 3000, 3500]
        # mahalanobis_N_range = [20, 40, 60,80, 100,120]
        if_N_range = np.sort(mahalanobis_N_range * 5)
        N_range = np.sort(mahalanobis_N_range * 10)
        print(np.shape(X))
        # normalize
        # from sklearn.preprocessing import Normalizer
        # transformer = Normalizer().fit(X)
        # X = transformer.transform(X)

    elif name == 'Http':
        # [0.81167218 0.10084469 0.08748314]
        # Good
        import hdf5storage
        mat = hdf5storage.loadmat(f'{data_dir}http.mat')
        X = mat['X']
        y = mat['y']
        print(len(y))
        print(np.sum(y))
        print(np.sum(y) / len(y))
        lof_krange = list(range(10, 110, 10)) * 6
        knn_krange = list(range(10, 110, 10)) * 6
        if_range = [0.5, 0.6, 0.7, 0.8, 0.9] * 6
        # mahalanobis_N_range=[1500, 2000, 2500, 3000, 3500, 4000]
        mahalanobis_N_range = [5000, 10000, 15000, 20000, 25000, 30000]
        # mahalanobis_N_range=[10000, 15000, 20000, 25000, 30000, 35000]
        # mahalanobis_N_range = [20, 40, 60,80, 100,120]
        if_N_range = np.sort(mahalanobis_N_range * 5)
        N_range = np.sort(mahalanobis_N_range * 10)

        # # remove duplicates
        # newdata = pd.DataFrame(np.concatenate((X,y), axis = 1)).drop_duplicates()
        # X = newdata[[0,1,2]].values
        # y = np.array([1 if i==1.0 else 0 for i in newdata[[3]].values])
        # print('Remove duplicates: ', len(y))

        # # normalize
        # from sklearn.preprocessing import Normalizer
        # transformer = Normalizer().fit(X)
        # X = transformer.transform(X)

    # elif name == 'Mulcross':
    #     pass

    elif name == 'Annthyroid':
        # Good
        # [0.30686872 0.1602263  0.09127541 0.08518756 0.06812179]
        filename = f'{data_dir}Annthyroid/Annthyroid_withoutdupl_norm_07.arff'
        X, y = load_dataset(filename=filename)
        print(np.shape(X), np.shape(y))
        print(sum(y) / len(y))
        N = 534
        num_outliers = [N, N, N, N]
        class_balance = [1 - N / 7129.0, N / 7129.0]
        lof_krange = list(range(10, 110, 10)) * 6
        knn_krange = list(range(10, 110, 10)) * 6
        if_range = [0.5, 0.6, 0.7, 0.8, 0.9] * 6
        # mahalanobis_N_range=[N]
        mahalanobis_N_range = [300, 400, 500, 600, 700, 800]
        if_N_range = np.sort(mahalanobis_N_range * 5)
        N_range = np.sort(mahalanobis_N_range * 10)

    elif name == 'Musk':
        # Good
        # [0.50269984 0.06793196 0.04755742 0.04595905 0.03887756]
        import hdf5storage
        mat = hdf5storage.loadmat(f'{data_dir}musk.mat')
        X = mat['X']
        y = mat['y']
        print(len(y))
        print(np.sum(y))
        print(np.sum(y) / len(y))
        lof_krange = list(range(10, 110, 10)) * 6
        knn_krange = list(range(10, 110, 10)) * 6
        if_range = [0.5, 0.6, 0.7, 0.8, 0.9] * 6
        mahalanobis_N_range = [100, 120, 140, 160, 180, 200]
        # mahalanobis_N_range = [20, 40, 60,80, 100,120]
        if_N_range = np.sort(mahalanobis_N_range * 5)
        N_range = np.sort(mahalanobis_N_range * 10)
        print(np.shape(X))

    elif name == 'Satimage-2':
        # [0.69531941 0.1529895  0.03707588 0.03439687 0.02486027]
        # Good
        import hdf5storage
        mat = hdf5storage.loadmat(f'{data_dir}satimage-2.mat')
        X = mat['X']
        y = mat['y']
        print(len(y))
        print(np.sum(y))
        print(np.sum(y) / len(y))
        lof_krange = list(range(10, 110, 10)) * 6
        knn_krange = list(range(10, 110, 10)) * 6
        if_range = [0.5, 0.6, 0.7, 0.8, 0.9] * 6
        mahalanobis_N_range = [60, 80, 100, 120, 140, 160]
        # mahalanobis_N_range = [20, 40, 60,80, 100,120]
        if_N_range = np.sort(mahalanobis_N_range * 5)
        N_range = np.sort(mahalanobis_N_range * 10)
        print(np.shape(X))

    elif name == 'Pendigits':
        # Bad
        # [0.29610985 0.22372539 0.14302142 0.09999411 0.06028273]
        import hdf5storage
        mat = hdf5storage.loadmat(f'{data_dir}pendigits.mat')
        X = mat['X']
        y = mat['y']
        print(len(y))
        print(np.sum(y))
        print(np.sum(y) / len(y))
        lof_krange = list(range(10, 110, 10)) * 6
        knn_krange = list(range(10, 110, 10)) * 6
        if_range = [0.5, 0.6, 0.7, 0.8, 0.9] * 6
        mahalanobis_N_range = [150, 200, 250, 300, 350, 400]
        # mahalanobis_N_range = [20, 40, 60,80, 100,120]
        if_N_range = np.sort(mahalanobis_N_range * 5)
        N_range = np.sort(mahalanobis_N_range * 10)
        print(np.shape(X))

    elif name == 'Mammography':
        # Bad
        # [0.40501236 0.20818947 0.16588271 0.10543556 0.09531255]
        import hdf5storage
        mat = hdf5storage.loadmat(f'{data_dir}mammography.mat')
        X = mat['X']
        y = mat['y']
        print(len(y))
        print(np.sum(y))
        print(np.sum(y) / len(y))
        lof_krange = list(range(10, 110, 10)) * 6
        knn_krange = list(range(10, 110, 10)) * 6
        if_range = [0.5, 0.6, 0.7, 0.8, 0.9] * 6
        mahalanobis_N_range = [150, 300, 500, 600, 900, 1000]
        # mahalanobis_N_range = [20, 40, 60,80, 100,120]
        if_N_range = np.sort(mahalanobis_N_range * 5)
        N_range = np.sort(mahalanobis_N_range * 10)
        print(np.shape(X))

    elif name == 'satellite':
        import hdf5storage
        mat = hdf5storage.loadmat(f'{data_dir}satellite.mat')
        X = mat['X']
        y = mat['y']
        print(len(y))
        print(np.sum(y))
        print(np.sum(y) / len(y))
        lof_krange = list(range(10, 110, 10)) * 6
        knn_krange = list(range(10, 110, 10)) * 6
        if_range = [0.5, 0.6, 0.7, 0.8, 0.9] * 6
        mahalanobis_N_range = [150, 300, 500, 600, 900, 1000]
        # mahalanobis_N_range = [20, 40, 60,80, 100,120]
        if_N_range = np.sort(mahalanobis_N_range * 5)
        N_range = np.sort(mahalanobis_N_range * 10)
        print(np.shape(X))

    elif name == 'cover':
        import hdf5storage
        mat = hdf5storage.loadmat(f'{data_dir}cover.mat')
        X = mat['X']
        y = mat['y']
        print(len(y))
        print(np.sum(y))
        print(np.sum(y) / len(y))
        lof_krange = list(range(10, 110, 10)) * 6
        knn_krange = list(range(10, 110, 10)) * 6
        if_range = [0.5, 0.6, 0.7, 0.8, 0.9] * 6
        mahalanobis_N_range = [150, 300, 500, 600, 900, 1000]
        # mahalanobis_N_range = [20, 40, 60,80, 100,120]
        if_N_range = np.sort(mahalanobis_N_range * 5)
        N_range = np.sort(mahalanobis_N_range * 10)
        print(np.shape(X))


    elif name == 'KDDCup99':
        # Bad
        # [0.32026138 0.28972426 0.12317104 0.09837007 0.04957106]
        filename = f'{data_dir}KDDCup99/KDDCup99_withoutdupl_norm_catremoved.arff'
        X, y = load_dataset(filename=filename)
        print(np.shape(X), np.shape(y))
        print(sum(y))
        N = 200
        num_outliers = [N, N, N, N]
        class_balance = [1 - N / 48113.0, N / 48113.0]
        lof_krange = list(range(10, 110, 10)) * 6
        knn_krange = list(range(10, 110, 10)) * 6
        if_range = [0.5, 0.6, 0.7, 0.8, 0.9] * 6
        mahalanobis_N_range = [500, 1000, 1500, 2000, 2500, 3000]
        # mahalanobis_N_range = [20, 40, 60,80, 100,120]
        if_N_range = np.sort(mahalanobis_N_range * 5)
        N_range = np.sort(mahalanobis_N_range * 10)

    elif name == 'ALOI':
        # Bad
        # [0.32026138 0.28972426 0.12317104 0.09837007 0.04957106]
        filename = f'{data_dir}ALOI/ALOI_withoutdupl_norm.arff'
        X, y = load_dataset(filename=filename)
        print(np.shape(X), np.shape(y))
        print(sum(y))
        N = 200
        num_outliers = [N, N, N, N]
        class_balance = [1 - N / 48113.0, N / 48113.0]
        lof_krange = list(range(10, 110, 10)) * 6
        knn_krange = list(range(10, 110, 10)) * 6
        if_range = [0.5, 0.6, 0.7, 0.8, 0.9] * 6
        mahalanobis_N_range = [500, 1000, 1500, 2000, 2500, 3000]
        # mahalanobis_N_range = [20, 40, 60,80, 100,120]
        if_N_range = np.sort(mahalanobis_N_range * 5)
        N_range = np.sort(mahalanobis_N_range * 10)

    elif "Friday" in name or "Thursday" in name or "Wednesday" in name:
        filename = f'{data_dir}{name}-2018_processed.csv'
        data = pd.read_csv(filename)
        y = data['label'].values
        X = data.drop(columns=['id', 'label'])

        # filename = f"{data_dir}{name}-2018_TrafficForML_CICFlowMeter.csv"
        # data = pd.read_csv(filename)
        # data = data[['Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts',
        #              'TotLen Fwd Pkts', 'TotLen Bwd Pkts', 'Fwd Pkt Len Max',
        #              'Fwd Pkt Len Min', 'Fwd Pkt Len Mean', 'Fwd Pkt Len Std',
        #              'Bwd Pkt Len Max', 'Bwd Pkt Len Min', 'Bwd Pkt Len Mean',
        #              'Bwd Pkt Len Std', 'Flow Byts/s', 'Flow Pkts/s', 'Flow IAT Mean',
        #              'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Tot',
        #              'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min',
        #              'Bwd IAT Tot', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max',
        #              'Bwd IAT Min', 'Fwd PSH Flags', 'Fwd Header Len', 'Bwd Header Len',
        #              'Fwd Pkts/s', 'Bwd Pkts/s', 'Pkt Len Min', 'Pkt Len Max',
        #              'Pkt Len Mean', 'Pkt Len Std', 'Pkt Len Var', 'FIN Flag Cnt',
        #              'SYN Flag Cnt', 'RST Flag Cnt', 'PSH Flag Cnt', 'ACK Flag Cnt',
        #              'URG Flag Cnt', 'ECE Flag Cnt', 'Down/Up Ratio', 'Pkt Size Avg',
        #              'Fwd Seg Size Avg', 'Bwd Seg Size Avg', 'Subflow Fwd Pkts',
        #              'Subflow Fwd Byts', 'Subflow Bwd Pkts', 'Subflow Bwd Byts',
        #              'Init Fwd Win Byts', 'Init Bwd Win Byts', 'Fwd Act Data Pkts',
        #              'Fwd Seg Size Min', 'Active Mean', 'Active Std', 'Active Max',
        #              'Active Min', 'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min', 'Label']].dropna()
        #
        # data = data.drop(data[data['Flow Duration'] == 'Flow Duration'].index)
        # try:
        #     data = data.drop(data.iloc[np.unique(np.where(np.array(data)[:, :-1] > 10000000000)[0])].index)
        # except:
        #     for col in data.columns[:-1]:
        #         data[col] = data[col].apply(lambda x: float(x))
        #     data = data.drop(data.iloc[np.unique(np.where(np.array(data)[:, :-1] > 10000000000)[0])].index)
        # X = data.drop(columns=['Label'])
        # y = data['Label'].apply(lambda x: {"Benign": 0}.get(x, 1)).values
        print(np.shape(X), np.shape(y))
        print(sum(y))
        N = sum(y)
        lof_krange = list(range(10, 100, 10)) * 6
        knn_krange = list(range(10, 110, 10)) * 6
        if_range = [0.5, 0.6, 0.7, 0.8, 0.9] * 6
        mahalanobis_N_range = [5000, 10000, 15000, 20000, 25000, 30000]
        # mahalanobis_N_range = [20, 40, 60,80, 100,120]
        if_N_range = np.sort(mahalanobis_N_range * 5)
        N_range = np.sort(mahalanobis_N_range * 10)


    else:
        print(f"{name} not recognized")
        raise NotImplementedError

    y = y.reshape(-1)
    return X, y, lof_krange, N_range, knn_krange, if_range, mahalanobis_N_range