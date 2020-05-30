# Import all the libraries
import pandas as pd
import numpy as np
import geopandas as gpd

from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier
import datetime
import warnings


# 1.- Write a function to read the data:
def reading_data(data):
    '''Read a csv and allocate in a df dataframe'''
    df = pd.read_csv(data)
    return df


# 2.- Write two functions to read analyze the data:
def get_info(df):
    '''Get the info about the columns of df dataframe'''
    return df.info()

def describe(df):
    '''Describe numerical the columns of df dataframe with principal stats'''
    return df.describe()

def get_sample(df):
    '''Return a sample from the dataframe df'''
    return df.sample(frac=0.02, replace=False, random_state=0)


# 3.- Create Training and Testing Sets
def split(df):
    '''Split the Dataset in train and test data.
    Inputs:
        df: dataframe with the information
    Output: df_train and df_test dataframes'''
    print('Total data before split:', df.shape[0])
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=0)
    print('Train size data:', df_train.shape[0])
    print('Test size data:', df_test.shape[0])
    return df_train, df_test

# 4.- Pre-Process Data (Two functions):
def replace_missing(df_train, df_test, continuous):
    '''Impute missing values of continuous variables using the median value.
    Inputs:
        df_train: dataframe with the train information
        df_test: dataframe with the test information
        continuous(list): List with the continuous columns on the dataframe
    '''
    d = {}
    print('\033[4m"Train data before fillna with median value:"\n\x1b[0m', df_train[continuous].isna().sum())
    print('\033[4m"Test data before fillna with median value:"\n\x1b[0m', df_test[continuous].isna().sum())
    for i in continuous:
        d[i] = df_train[i].median()
    print('\n\033[1m\33[92m"Median Values to fill"', d, '\x1b[0m\n')
    df_train = df_train.fillna(value=d)
    df_test = df_test.fillna(value=d)
    print('\033[4m\033[94m"Sanity check: Train data after fillna with median value"\n\x1b[0m', df_train[continuous].isna().sum())
    print('\033[4m\033[94m"Sanity check: Test data after fillna with median value"\n\x1b[0m', df_test[continuous].isna().sum())
    return df_train, df_test


def do_normalization(df_train, df_test, continuous):
    '''Normalize continuous variables on the dataframe
       Inputs:
           df_train: dataframe with the train information
           df_test: dataframe with the test information
           continuous(list): List with the continuous columns on the dataframe df
    '''
    for i in continuous:
        print('Normalizing column:', i)
        scaler = StandardScaler().fit(df_train[[i]])
        df_train[i] = scaler.transform(df_train[[i]])
        df_test[i] = scaler.transform(df_test[[i]])
    print('\033[4m\033[94m"Sanity check: Mean ans std for train set:"\n\x1b[0m', df_train[continuous].describe().loc[['mean', 'std']])
    print('\033[4m\033[94m"Sanity check: Mean ans std for test set:"\n\x1b[0m',df_test[continuous].describe().loc[['mean', 'std']])
    return df_train, df_test


# 5.- Generate Features
def one_hot(df_train, df_test, categorical):
    '''Get dummies of categorical variables on the dataframe
       Create columns missing on test dataframe after the dummies process
       Drop additional columns on test dataframe after the dummies process
       Inputs:
           df_train: dataframe with the train information
           df_test: dataframe with the test information
           categorical(list): List with the categorical columns on the dataframe df
    '''
    print('Get dummy: df_train columns...', )
    df_train = pd.get_dummies(df_train, columns=categorical)
    print('Get dummy: df_test columns...', )
    df_test = pd.get_dummies(df_test, columns=categorical)
    
    #Check missing columns on the test dataset
    no_test = set(df_train.columns) - set(df_test.columns)
    if len(no_test) > 0:
        print('\033[94m"Sanity check1: Missing columns to create on test dataframe:"\x1b[0m', no_test)
        #Create the missing columns with 0 on the test dataset
        for i in no_test:
            print('Creating column:', i, 'on df_test.')
            df_test[i] = 0
    else:
        print('\033[94m"Sanity check1: No missing columns on test dataframe."\n\x1b[0m')

    #Check additional columns on the test dataset
    
    no_train = set(df_test.columns) - set(df_train.columns )
    if len(no_train) > 0:
        print('\033[94m"Sanity check2: Additional columns to drop on test dataframe:"\x1b[0m', no_train)
        #Drop that columns
        df_test = df_test.drop(no_train, axis=1)
    else:
        print('\033[94m"Sanity check2: No additional columns to drop on test dataframe"\n\x1b[0m')
    df_test = df_test[df_train.columns]
    return df_train, df_test


def discretize(df, categorical, bins):
    '''Discretize categorical variables on the dataframe
       Inputs:
           df : dataframe with the information
           categorical(list): List with the categorical columns on the dataframe df
           bins = number of bins for the categorization
    '''
    for i in categorical:
        df[i] = pd.cut(df[i], bins)
    return df


# 6.- Build Classifiers
def classification(train_target, test_target, train_features, test_features, MODELS, GRID):
    '''Applies machine learning model(s) to a dataset and returns two dataframes.
       Inputs:
           train_target: dataframe with the train target information
           test_target: dataframe with the test target information
           train_features: dataframe with the train features
           test_features: dataframe with the test features
           MODELS(dict) :  dictionary with the models
           GRID(dict) :  dictionary with the parameters of the models
       Outputs:
           results: dataframe with the models and their accuracy information
           coeff: dataframe with each absolute coeff for each model run
    '''

    # Begin timer 
    start = datetime.datetime.now()
    
    # Initialize results data frame 
    results = pd.DataFrame()

    # Loop over models 
    for model_key in MODELS.keys(): 
        
        # Loop over parameters 
        for params in GRID[model_key]: 
            print("Training model:", model_key, "|", params)
            
            # Create model
            d = {}
            model = MODELS[model_key]
            model = model.set_params(**params)
            d['model'] = model
            # Begin timer for the model
            start_model = datetime.datetime.now()

            # Fit model on training set 
            model.fit(train_features,train_target)

            # Predict on testing set 
            target_predicted = model.predict(test_features)
            
            #Cross Validation
            k = 10
            kf = KFold(n_splits=k, shuffle=True, random_state=0)

            cv_results = cross_val_score(model,
                             train_features,
                             train_target,
                             cv=kf,
                             scoring='accuracy')
        
            # Evaluate predictions

            accuracy, precision, recall = metrics(test_target, target_predicted)
            end_model = datetime.datetime.now()
            
            # Store results in your results data frame
            d['Model']= model_key
            criterion: gini or entropy
            criterion = params.get('criterion', 'None')
            d['criterion'] = criterion
            max_depth = params.get('max_depth', 'None')
            d['max_depth'] = max_depth
            min_samples_split = params.get('min_samples_split', 'None')
            d['min_samples_split'] = min_samples_split
            d['Accuracy'] = round(accuracy * 100, 4)
            d['precision'] = round(precision * 100, 4)
            d['recall'] = round(recall * 100, 4)
            d['Time_model'] = (end_model - start_model)
            d['Cross_validation'] = cv_results.mean()
            d['feature_importance'] = model.feature_importances_
            results = results.append(d, ignore_index=True)
       
    # End timer
    stop = datetime.datetime.now()
    print("\033[91m\033[1m'Time Elapsed:", stop - start, '\033[0m')
    return results


# 7. Evaluate Classifiers
#Define a new function for Accuracy metrics
def metrics(test_target, target_predicted):

    accuracy = accuracy_score(test_target, target_predicted)
    precision = precision_score(test_target,target_predicted)
    recall = recall_score(test_target,target_predicted)

    return accuracy, precision, recall
