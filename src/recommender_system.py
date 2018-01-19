import string
import graphlab
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

def clean_data(filename):
    '''Preprocess the jokes text
    Input:
        filename: jokes file path
    Returns:
        text: list containing 150 jokes
    '''
    with open(filename) as f:
        text = f.read().lower().split('</p>')
        text = [line.replace('\r', '') for line in text]
        text = [line.replace('\n', '') for line in text]
        text = [line.replace('<br />', '') for line in text]
        text = [line.replace('<p>', '') for line in text]
        text = [line.replace('&quot;', '') for line in text]
        text = [line.replace('&#039;', '') for line in text]
        #text = [re.sub(" \d+", " ", line) for line in text]
        text = [line.split(':', 1)[-1] for line in text]
        #text = text[:150]
    return text

def load_train_test_data(filename_train, filename_test):
    ''' Load training and test data
    Input:
        filename_train: path of train data
        filename_test: path of test data
    Returns:
        train_data: dataframe containing train data
        test_data: dataframe containing test data
    '''
    user_ratings_train = pd.read_csv(filename_train, sep='\t')
    user_ratings_test = pd.read_csv(filename_test, sep=',')
    train_data = graphlab.SFrame(user_ratings_train)
    test_data = graphlab.SFrame(user_ratings_test)
    return train_data, test_data

def matrix_factorization_model(data, num_factors):
    ''' Build NMF model using graphlab
    Input:
        data: dataframe containing rating data
        num_factors: number of latent features
    Returns:
        mf_model: graphlab NMF model instance
    '''
    mf_model = graphlab.recommender.factorization_recommender.create(data,target
    ='rating', user_id = 'user_id', item_id = 'joke_id', num_factors=num_factors)
    return mf_model

def get_model_results(mf_model, data):
    ''' Get predicted ratings, user and joke factors matrices
    Input:
        mf_model: graphlab NMF model
        data: dataframe containing rating data
    Output:
        user_factors: dataframe containing the user factors
        joke_factors: dataframe containing the joke factors
    '''
    predicted_rating = mf_model.predict(data)
    coeffs = mf_model.get('coefficients')
    user_factors = coeffs['user_id']['factors'].to_numpy()
    joke_factors = coeffs['joke_id']['factors'].to_numpy().T
    jokes =  coeffs['joke_id']['joke_id']
    users = coeffs['user_id']['user_id']
    pred_rating_matrix_multiply = np.dot(user_factors,joke_factors) + coeffs['intercept']
    print('predicted rating: {}'.format(predicted_rating))
    print('user factors: {}'.format(user_factors))
    print('joke factors: {}'.format(joke_factors))
    return user_factors, joke_factors

def grid_search_mf_model(data, params, num_folds = 5):
    ''' Grid search NMF model for the best parameters
    Input:
        data: dataframe containing rating data
        params: dictionary containing the model parameters
        num_folds: number of folds
    Returns:
        
    '''
    folds = graphlab.cross_validation.KFold(data, num_folds)
    job = graphlab.grid_search.create(folds,
        graphlab.recommender.factorization_recommender.create,params)
    print job.get_results()

def test_rating_rmse(mf_model, test_data):
    ''' Predict ratings on test data and get RMSE score
    Input:
        mf_model: graphlab NMF model
        test_data: dataframe containing test rating data
    Returns:
        predicted_test_rating: numpy array containing predicted ratings
        test_rmse: rmse score, float
    '''
    predicted_test_rating = mf_model.predict(test_data)
    test_rmse = np.sqrt(mean_squared_error(predicted_test_rating, test_data['rating']))
    return predicted_test_rating, test_rmse

def find_latent_features(text, user_factors, joke_factors):
    ''' Find the latent topics or categories in the jokes data
    Input:
        text: list contaning all the jokes
        user_factors: dataframe containing user factors
        joke_factors: dataframe containing joke factors
    Returns:
        
    '''
    joke_factors_sorted = np.argsort(joke_factors, axis = 1)[:,-10:]
    for i in range(user_factors.shape[1]):
        print [text[i-1] for i in joke_factors_sorted[i]][::-1]

if __name__ == '__main__':
    '''Load clean jokes text data and user ratings
    '''
    text = clean_data('../data/jokes.dat')
    train_data, test_data = load_train_test_data('../data/ratings.csv',
                                                '../data/test_ratings.csv')

    '''Grid search for the best parameters
    '''
    params = {'user_id':'user_id', 'item_id':'joke_id', 'target':'rating',
                'num_factors': [2, 4, 6, 8]}
    grid_search_mf_model(train_data, params, num_folds = 5)

    '''Build matrix factorization model
    '''
    mf_model = matrix_factorization_model(train_data, 8)
    user_factors, joke_factors = get_model_results(mf_model, train_data)

    '''Predict the ratings on the test data
    '''
    predicted_test_rating, test_rmse = test_rating_rmse(mf_model, test_data)
    print('predicted test rating: {}'.format(predicted_test_rating))
    print('test rmse: {}'.format(test_rmse))

    '''Find the latent features in the jokes
    '''
    find_latent_features(text, user_factors, joke_factors)


