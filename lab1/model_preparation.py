import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM


def create_features_and_target(arr: np.ndarray, look_back=1) -> tuple:
    dataX, dataY = [], []
    
    for i in range(len(arr)-look_back-1):
        dataX.append(arr[i:(i+look_back), 0])
        dataY.append(arr[i+look_back, 0])
    
    return np.array(dataX), np.array(dataY)


def main()-> None:
    look_back = 1

    #preparing train data
    train_dataset_path = 'after_preprocessing/train.npy'
    train = np.load(train_dataset_path)
    trainX, trainY = create_features_and_target(train, look_back)

    #midel fiting
    model = Sequential()
    model.add(LSTM(4, input_shape=(1, look_back)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)
    model.save('model.keras')


if __name__ == '__main__':
    main()
