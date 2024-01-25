# STAIR and L-STAIR
This is the implementaion for the paper **Interpretable Outlier Summarization**


## Requirements
python 3.7.0  
numpy 1.19.4  
scikit-learn 1.2.2  
tqdm 4.65.0  
corels 1.1.29  
wittgenstein 0.3.4  

## Datasets
All the datasets are provided [here](https://drive.google.com/drive/folders/1kINie6My69DxtM5aGtZiPpRX81qOkJRs?usp=sharing). you can download them and move them into the folder `data`. The content under the folder `data` should be:
```
-- data
    -- SpamBase_withoutdupl_norm_40.arff
    -- Pima_withoutdupl_norm_35.arff
    -- cover.mat
    -- mammography.mat
    -- PageBlocks_norm_10.arff
    -- pendigits.mat
    -- satellite.mat
    -- satimate-2.mat
    -- shuttle.mat
    -- Thursday-01-03-2018_processed.csv
    -- winequality-white.csv
```


## Cammands

### Outlier Detection Task
For Outlier Detection datasets, you need to first enter the `OutlierDetection` folder:
```
cd OutlierDetection
```
The following command will run the baselines **ID3**, **CART** and our algorithm **STAIR** successively on the dataset `PageBlock`.
```
python main.py PageBlock
```
If you need to run the algorithm **L-STAIR** on the dataset `PageBlock`, you can use the following command:
```
python lstair_main.py PageBlock
```
If you need to run the algorithms on other dataset, simply change the dataset name `PageBlock` into other names such as `Pendigits`, `Pima` and so on.

If you need to run the algorithm **Explanation Tbale** on the dataset `Pendigits`, you can use the following commands:
```
python explanation_table_main.py Pendigits
```
If you need to run the algorithms on other dataset, simply change the dataset name `Pendigits` into other names such as `PageBlock`, `Pima` and so on.

If you need to run the algorithm **RIPPERk** on the dataset `Pendigits`, you can use the following commands:
```
python ripperk_main.py Pendigits
```
If you need to run the algorithms on other dataset, simply change the dataset name `Pendigits` into other names such as `PageBlock`, `Pima` and so on.

If you need to run the algorithm **HiCS** on the dataset `Pendigits`, you can use the following commands:
```
git clone https://github.com/nim-lang/Nim.git
cd Nim
./build_all.sh (for linux)
HiCS -- csvIn Pendigits
```
If you need to run the algorithms on other dataset, simply change the dataset name `Pendigits` into other names such as `PageBlock`, `Pima` and so on.

If you need to run the algorithm **FRL** on the dataset `Pendigits`, you can use the following commands:
```
python frl_main.py Pendigits
```
If you need to run the algorithms on other dataset, simply change the dataset name `Pendigits` into other names such as `PageBlock`, `Pima` and so on.


If you need to run the algorithm **CORELS** on the dataset `Pendigits`, you can use the following commands:
```
python corels_main.py Pendigits
```
If you need to run the algorithms on other dataset, simply change the dataset name `Pendigits` into other names such as `PageBlock`, `Pima` and so on.



### MultiClass Classification Task
For Classification datasets, you need to first enter the `MultiClassClassification` folder:
```
cd MultiClassClassification
```
The following command will run the baselines **ID3**, **CART** and our algorithm **STAIR** successively on the dataset `Wine`.
```
python main.py Wine
```
The command to run **L-STAIR** on the dataset `Wine` is:
```
python lstair_main.py Wine
```

