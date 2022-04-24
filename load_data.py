import pandas as pd
from scipy import stats
import numpy as np


def _remove_outlier_z(data, th=3):
    """Remove outliers using the z-scoer method

    Args:
        data (_type_): pandas DataFrame 
        th (int, optional): Threshold value of the normal distribution (sigma). Defaults to 3.

    Returns:
        cleaned DataFrame 
    """
    # s1 = data.shape
    z_score = np.abs(stats.zscore(data))
    idx = np.where(z_score>th)
    data = data.drop(idx[0], axis=0)
    data = data.dropna()
    # s2 = data.shape
    # print(f'Shape before removing outliser {s1} \t\t shape after removing the outliers {s2}')
    
    return data        


def load_and_preprocess(path1 = './data/well1.csv', path2='./data/well2.csv'):
    """Read, load, and prepared test and train data

    Args:
        path1 (str, optional): path to well 1 data (csv file). Defaults to './data/well1.csv'.
        path2 (str, optional): path to well 2 data (csv file). Defaults to './data/well2.csv'.

    Returns:
        tuple of train and test data
    """
    w1 = pd.read_csv(path1)
    w2 = pd.read_csv(path2)

    w1 = _remove_outlier_z(w1, th=3)
    w2 = _remove_outlier_z(w2, th=3)
    
    w1 = w1.reset_index().drop(['index'], axis=1)
    w2 = w2.reset_index().drop(['index'], axis=1)
    
    w1 = w1[w1>0].dropna()
    w2 = w2[w2>0].dropna()
    
    w1 = w1.sort_values(by=['Hole Depth'], ascending=True)
    w2 = w2.sort_values(by=['Hole Depth'], ascending=True)
    
    w2_dummy = w2.copy()
    w2_dummy = w2_dummy[w2_dummy['Hole Depth']<9000]
    test_sectoin = w2[w2['Hole Depth']>8999]
    
    df = pd.concat([w1, w2_dummy], axis=0)
    
    drop = ['Standpipe Pressure', 'Bit Depth', 'Bit RPM', 'Min RPM', 'Min Hook Load', 'Min Pressure', 
        'Line Wear', 'Block Height', 'Time Of Penetration', 'Min WOB']

    df = df.drop(drop, axis=1)
    test_sectoin = test_sectoin.drop(drop, axis=1)
    
    del w1, w2, w2_dummy
    
    X_train = df.drop(['Rate Of Penetration'], axis=1).to_numpy()
    y_train = df['Rate Of Penetration'].to_numpy()


    X_test = test_sectoin.drop(['Rate Of Penetration'], axis=1).to_numpy()
    y_test = test_sectoin['Rate Of Penetration'].to_numpy()
    
    return (X_train, y_train), (X_test, y_test)
    


