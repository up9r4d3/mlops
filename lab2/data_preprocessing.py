import numpy as np
import pandas as pd
import pickle

from sklearn.preprocessing import MinMaxScaler
from data_creation import check_folder


def create_processed_data(input_path1: str, input_path2: str, 
                    dataset_name, output_folder: str) -> None:
    #read samples
    df1 = pd.read_csv(input_path1, index_col=0)
    df2 = pd.read_csv(input_path2, index_col=0)
    
    #merge samples
    df = (
        df1
        .merge(df2, how='inner', on='Date')
        .sort_values(by='Date', ascending=True)
    )

    #normalize data
    scaler = MinMaxScaler()
    temps_arr = ((df.iloc[:, 1]+df.iloc[:, 2])/2).values.reshape(-1, 1)
    temps_arr = scaler.fit_transform(temps_arr)
    temps_arr = temps_arr.astype('float32')
    
    #save data
    np.save(f'{output_folder}{dataset_name}.npy', temps_arr)

    #save scaler model
    model_folder = 'models/'
    check_folder(model_folder)
    with open(f'{model_folder}scaler.pkl', 'wb') as output:
        pickle.dump(scaler, output)


def main() -> None:
    train_input_path1 = 'train/train_sample1.csv'
    train_input_path2 = 'train/train_sample2.csv'

    test_input_path1 = 'test/test_sample1.csv'
    test_input_path2 = 'test/test_sample2.csv'

    create_processed_data(train_input_path1, train_input_path2, 'train', 'train/')
    create_processed_data(test_input_path1, test_input_path2, 'test', 'test/')


if __name__ == '__main__':
    main()
