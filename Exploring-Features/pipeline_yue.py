# Yue Kuang
# May 13th 2020
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn import metrics
import datetime


#Read Data
def read_data(filename):
    '''
    read data into pandas dataframe
    '''
    df =  pd.read_csv(filename)
    return df

#Explore Data
def explore_corr(df):
    '''
    produce a correlation matrix between numeric features in the dataframe
    '''
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numeric_df = df.select_dtypes(include=numerics)
    corr_matrix = numeric_df.corr()
    return corr_matrix

def explore_outlier(df):
    '''
    find outliers in a dataframe (only for numeric columns)
    '''
    #only check outliers for numeric columns
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numeric_df = df.select_dtypes(include=numerics)  
    outliers_df = numeric_df[(np.abs(stats.zscore(numeric_df)) > 3).all(axis=1)]
    return outliers_df

def train_test(df):
    '''
    split dataframe into training and testing set
    '''
    train, test = train_test_split(df, test_size=0.2, random_state=0)
    return train, test
 
def normalize(train_data, test_data ,features):
    '''
    normalize training data and testing data's features
    '''
    #features are only continuous variables
    if len(features) >1:
        train_features = train_data[features]
        test_features = test_data[features]
    elif len(features) == 1:
        train_features = train_data[features].values.reshape(-1,1)
        test_features = test_data[features].values.reshape(-1,1)
    
    #normalize
    scaler = StandardScaler()
    norm_train_features = scaler.fit_transform(train_features)
    #manually normalize test data
    norm_test_features= (test_features- train_features.mean()) / train_features.std(ddof=0)

    return norm_train_features, norm_test_features

def fix_missing(train, test, features):
    '''
    fill specific features's missling values with median values in training and testing data
    '''
    #features are only continuous variables
    data_list = [train, test]
    value_dict = {}
    for feature in features:
        median = train[feature].median()
        value_dict[feature] = median
    for df in data_list:
        df.fillna(value=value_dict, inplace = True)
    
    return train, test
        
#generate features
def encode_cat(train, test, features):
    '''
    one-hot-encoding categorical features in training and testing data
    '''
    for feature in features:
        train_cat = train[feature].unique().tolist()
        test_cat =  test[feature].unique().tolist()
        in_tr_notin_test = list(set(train_cat)-set(test_cat))
        in_test_notin_tr = list(set(test_cat)-set(train_cat))
        
        #transform training data
        train_dummy = pd.get_dummies(train[feature],prefix = feature)
        train = train.drop(feature, axis=1)
        train =  train.join(train_dummy)
        # If a value appears in your testing set but not your 
        #training set, drop it from your testing data.
        for cat in in_test_notin_tr:
            index = test[test[feature]==cat].index
            test = test.drop(index)

        test_dummy = pd.get_dummies(test[feature], prefix = feature)
        test = test.drop(feature, axis=1)
        test = test.join(test_dummy)

        if len(in_tr_notin_test) != 0:
            for cat in in_tr_notin_test:
                new_col =  str(feature)+"_"+str(cat)
                test[new_col] = 0
    return train, test

def bucket_con(bucket_dict, train, test):
    '''
    discretizing continuous variables
    
    Input: 
    bucket_dict: (dict) a dictionary containing info of what features to discretize and how to discretize.
                    Example:{feature1:bin, feature2: bin}
    train: training dataframe
    test: test dataframe
    '''

    df_list = [train, test]
    for feature, bin in bucket_dict.items():
        for df in df_list:
            df[feature] = pd.cut(df[features], bin)

    return train, test

def get_target(train_data, test_data,target):
    train_target = train_data[target].values.reshape(-1)
    test_target = test_data[target].values.reshape(-1)
    return train_target, test_target

#Build Classifiers
def grid_search(tr_feature, t_feature,tr_target,t_target, MODELS, GRID ):
    '''
    train a series of models with various parameters
    return: a dataframe containing evaluation info of all models
    '''
    # Begin timer 
    start = datetime.datetime.now()

    # Initialize results data frame 
    row_list = []

    # Loop over models 
    for model_key in MODELS.keys(): 
        # Loop over parameters 
        for params in GRID[model_key]: 
            dict1={}
            print("Training model:", model_key, "|", params)
            print(params)
            dict1["training_model"] = model_key
            dict1["parameters"] = params
            # Create model 
            model = MODELS[model_key]
            model.set_params(**params)
            
            # Fit model on training set 
            model.fit(tr_feature, tr_target)
            
            # Predict on testing set 
            target_predict =  model.predict(t_feature)
            
            # Evaluate predictions 
            accuracy_score = metrics.accuracy_score(t_target, target_predict)

            # Store results in your results data frame 
            print(accuracy_score)
            dict1["accuracy_score"] = accuracy_score
            row_list.append(dict1)

    # End timer
    stop = datetime.datetime.now()
    print("Time Elapsed:", stop - start) 
    eval_df = pd.DataFrame(row_list)
    return eval_df
#Evaluate Classifiers

def eval_class(test_target, target_predict):
    '''
    calculate accuracy score for a model
    '''
    acc=np.mean(test_target==target_predict)
    return acc
