import numpy as np
import pickle
import keras

from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from model_preparation import create_features_and_target


def main() -> None:
    #preparing test data
    test_dataset_path = 'test/test.npy'
    test = np.load(test_dataset_path)
    testX, testY = create_features_and_target(test, look_back=1)

    #load fitted model
    model = keras.models.load_model('models/model.keras')
    
    #load fitted scaler
    with open('models/scaler.pkl', 'rb') as pkl_file_1:
        scaler = pickle.load(pkl_file_1)
    
    # make predictions
    testPredict = model.predict(testX)
    
    # invert predictions
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])
    
    # calculate root mean squared error
    RMSE_Score = np.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
    print('Test Score: %.2f RMSE' % (RMSE_Score))
    MAE_Score = mean_absolute_error(testY[0], testPredict[:,0])
    print('Test Score: %.2f MAE' % (MAE_Score))
    MPAE_Score = mean_absolute_percentage_error(testY[0], testPredict[:,0])
    print('Test Score: %.2f MPAE' % (MPAE_Score))


if __name__ == '__main__':
    main()
