import pandas as pd
import os

from sklearn.model_selection import train_test_split


def check_folder(folder: str) -> None:
    if not os.path.exists(folder):
        os.makedirs(folder)


def main() -> None:
    #reading datasets
    df1 = pd.read_csv('data/daily-min-temperatures.csv')
    df2 = pd.read_csv('data/daily-max-temperatures.csv')

    #dividing to samples
    train_sample_1, test_sample_1 = train_test_split(df1, test_size=0.3, random_state=13)
    train_sample_2, test_sample_2 = train_test_split(df2, test_size=0.3, random_state=13)

    # save samples
    folder = 'train/'
    check_folder(folder)
    train_sample_1.to_csv(f'{folder}train_sample1.csv')
    train_sample_2.to_csv(f'{folder}train_sample2.csv')
    
    folder = 'test/'
    check_folder(folder)
    test_sample_1.to_csv(f'{folder}test_sample1.csv')
    test_sample_2.to_csv(f'{folder}test_sample2.csv')


if __name__ == '__main__':
    main()
