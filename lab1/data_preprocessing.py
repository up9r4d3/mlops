import numpy as np
import pandas as pd
import pickle

from sklearn.preprocessing import MinMaxScaler
from data_creation import check_folder


def create_dataset(input_path_1: str, input_path_2: str, 
                    dataset_name, output_path: str) -> None:
    #read samples
    df1 = pd.read_csv(input_path_1, index_col=0)
    df1 = df1.sort_values(by='Date', ascending=True)

    df2 = pd.read_csv(input_path_2, index_col=0)
    df2 = df2.sort_values(by='Date', ascending=True)
    
    #merge samples
    df = df1.merge(df2, how='inner', on='Date')

    #normalize data
    scaler = MinMaxScaler()
    temps_arr = ((df.iloc[:, 1]+df.iloc[:, 2])/2).values.reshape(-1, 1)
    temps_arr = scaler.fit_transform(temps_arr)
    temps_arr = temps_arr.astype('float32')
    
    np.save(f'{output_path}{dataset_name}.npy', temps_arr)

    with open('scaler.pkl', 'wb') as output:
        pickle.dump(scaler, output)


def main() -> None:
    train_input_path_1 = 'train/train_sample1.csv'
    train_input_path_2 = 'train/train_sample2.csv'

    test_input_path_1 = 'test/test_sample1.csv'
    test_input_path_2 = 'test/test_sample2.csv'

    output_path = 'after_preprocessing/'
    check_folder(output_path)
    create_dataset(train_input_path_1, train_input_path_2, 'train', output_path)
    create_dataset(test_input_path_1, test_input_path_2, 'test', output_path)

if __name__ == '__main__':
    main()
