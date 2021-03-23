import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import datetime
import gc
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def com_rsq(target, predictions):
    """
    Computes r-squared
    """
    return sqrt(mean_squared_error(target, predictions))



def one_model_lgbm(x_train, x_valid, y_train, y_valid, params):
    """Fit one LGBM model
    """

    dtrain=lgb.Dataset(x_train, label=y_train)
    dvalid=lgb.Dataset(x_valid, label=y_valid)

    watchlist=dvalid

    booster=lgb.train(
        params=params,
        train_set=dtrain,
        valid_sets=watchlist,
        verbose_eval=20000
    )

    return booster


def compute_history(data):
    """
    Computes history time features for stock time-forecasting task
    """

    for history in [1,2,3,5,7,10,14,21,30, 45]:
        for feature in ['Open', 'High', 'Low', 'Close', 'Volume']:
            name_of_feature_laged = feature+'_'+str(history)
    
            data[name_of_feature_laged] = data[feature].rolling(history, min_periods=1).mean().reset_index(drop=True)
            data[name_of_feature_laged] = data[name_of_feature_laged].shift(-history)
            
    #Compute only shifted volumne by 2, 3:
    data['Volume_shifted_2'] = data['Volume'].shift(-2)
    data['Volume_shifted_3'] = data['Volume'].shift(-3)

    data['day']=pd.to_datetime(data['Date']).dt.day
    data['weekday']=pd.to_datetime(data['Date']).dt.dayofweek
    data['month']=pd.to_datetime(data['Date']).dt.month
    return data



def compute_fractions(data):
    """
    Computes ratios of historical features
    """

    for history in [2,3,5,7,10,14,21,30, 45]:
        for feature in ['Open', 'High', 'Low', 'Close', 'Volume']:
            name_of_feature = feature+'_yest_div_'+str(history)
            
            data[name_of_feature] = data[f"{feature}_1"]/data[f"{feature}_{history}"]

    return data


def graph_exploration_continuous(feature_binned,target):
    """
    Function that visualises relationship between given binned variable and 
    continuous target
    """


    if(sum(feature_binned.isnull())>0):
        feature_binned=feature_binned.cat.add_categories('NA').fillna('NA')
    
    plt.figure(figsize=(12,5))
    sns.boxplot(x=feature_binned,y=target,showfliers=False)
    plt.xticks(rotation='vertical')
    plt.show()


def one_model_rf(x_train, x_valid, y_train, y_valid):
    """Fit one RF model
    """

    # Firstly, we will try to find optimal numer of trees
    
    best_performance = 0
    opt_num_nn = 10
    
    for nn in [10,20,50,100, 200]:
           
        forest = RandomForestClassifier(n_estimators=nn, max_depth=3, random_state=0)
        forest.fit(x_train, y_train)
        
        score = r2_score(y_valid, forest.predict(x_valid))
        if score > best_performance:
            best_performance = score
            opt_num_nn = nn
            
    # Now we have optimal number of neigh
    
    forest = RandomForestClassifier(n_estimators = opt_num_nn, max_depth=3, random_state=0)
    forest.fit(x_train, y_train)

    return forest


def comp_var_imp(models,preds):
    """
    Computes variable importances
    """

    importance_df=pd.DataFrame()
    importance_df['Feature']=preds
    importance_df['Importance_gain']=0
    importance_df['Importance_weight']=0

    for model in models:
        importance_df['Importance_gain'] = importance_df['Importance_gain'] + model.feature_importance(importance_type = 'gain') / len(models)
        importance_df['Importance_weight'] = importance_df['Importance_weight'] + model.feature_importance(importance_type = 'split') / len(models)

    return importance_df



def plot_importance(models, imp_type, preds ,ret=False, show=True, n_predictors = 100):
    """Plots variable importances
    """
    if ((imp_type!= 'Importance_gain' ) & (imp_type != 'Importance_weight')):
        raise ValueError('Only importance_gain or importance_gain is accepted')

    dataframe = comp_var_imp(models, preds)

    if (show == True):
        plt.figure(figsize = (20, len(preds)/2))
        sns.barplot(x=imp_type, y='Feature', data=dataframe.sort_values(by=imp_type, ascending= False).head(len(preds)))

    if (ret == True):
        #return dataframe.sort_values(by=imp_type, ascending= False).head(len(preds))[['Feature', imp_type]]
        return dataframe.head(len(preds))[['Feature', imp_type]]



def train_model_CV(train, test, preds,  target, cross_val):
    """
    Train cross-validation model
    """

    params={'early_stopping_rounds': 100,
        'num_boost_round': 10000,
        'learning_rate': 0.01,
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'rmse',
        'subsample': 0.7,
        'max_depth': 3,
        'seed': 1234, 
        'verbosity': 0}

    predictions_lgbm = np.zeros(test.shape[0])
    predictions_rf = np.zeros(test.shape[0])
    
    models_lgbm = []
    models_rf = []

    train_lgbm_results = []
    valid_lgbm_results = []

    train_rf_results = []
    valid_rf_results = []

    
    for train_indexes, valid_indexes in cross_val.split(train):
        train_data_lgbm= train.iloc[train_indexes][preds]
        valid_data_lgbm = train.iloc[valid_indexes][preds]

    
        train_target = train[target].iloc[train_indexes]
        valid_target = train[target].iloc[valid_indexes]
        
        model = one_model_lgbm(train_data_lgbm, valid_data_lgbm, train_target, valid_target, params)

        models_lgbm.append(model)

        print(f'R-squared from LGBM on training data: {r2_score(train_target, model.predict(train_data_lgbm))}')
        print(f'R-squared from LGBM on validationdata: {r2_score(valid_target, model.predict(valid_data_lgbm))}')
        train_lgbm_results.append(r2_score(train_target, model.predict(train_data_lgbm)))
        valid_lgbm_results.append(r2_score(valid_target, model.predict(valid_data_lgbm)))

        model = one_model_rf(train_data_lgbm, valid_data_lgbm, train_target, valid_target)

        models_rf.append(model)

        print(f'R-squared from RF on training data: {r2_score(train_target, model.predict(train_data_lgbm))}')
        print(f'R-squared from RF on validationdata: {r2_score(valid_target, model.predict(valid_data_lgbm))}')
        train_rf_results.append(r2_score(train_target, model.predict(train_data_lgbm)))
        valid_rf_results.append(r2_score(valid_target, model.predict(valid_data_lgbm)))


        print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')

    for model in models_lgbm:
        predictions_lgbm = predictions_lgbm + model.predict(test[preds]) / len(models_lgbm)

    for model in models_rf:
        predictions_rf = predictions_rf + model.predict(test[preds]) / len(models_rf)


    importance = plot_importance(models_lgbm, imp_type = 'Importance_gain', preds = preds ,ret=True, show=False, n_predictors = len(preds))
    
    return [predictions_lgbm,
           predictions_rf,
           importance,
           train_lgbm_results,
           valid_lgbm_results,
           train_rf_results,
           valid_rf_results]