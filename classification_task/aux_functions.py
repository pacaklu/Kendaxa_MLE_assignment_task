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
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score




def graph_exploration(feature_binned,target):
    """
    Plot of relationship between target and categorical feature.
    """
  
    if(sum(feature_binned.isnull())>0):
        feature_binned=feature_binned.cat.add_categories('NA').fillna('NA')
    
    result = pd.concat([feature_binned, target], axis=1)
    
    gb=result.groupby(feature_binned)
    counts = gb.size().to_frame(name='counts')
    final=counts.join(gb.agg({result.columns[1]: 'mean'}).rename(columns={result.columns[1]: 'target_mean'})).reset_index()
    final['odds_ratio']=np.log2((final['counts']*final['target_mean']+100*np.mean(target))/((100+final['counts'])*np.mean(target)))
        
    sns.set(rc={'figure.figsize':(15,10)})
    fig, ax =plt.subplots(2,1)
    sns.countplot(x=feature_binned, hue=target, data=result,ax=ax[0])
    sns.barplot(x=final.columns[0],y='odds_ratio',data=final,color="green",ax=ax[1])
    plt.show()
    
    
def graph_exploration_continuous(feature_binned,target):
    """
    Plot of relationship between target and continuous feature.
    """
  
    if(sum(feature_binned.isnull())>0):
        feature_binned=feature_binned.cat.add_categories('NA').fillna('NA')
    
    plt.figure(figsize=(12,5))
    sns.boxplot(x=feature_binned,y=target,showfliers=False)
    plt.xticks(rotation='vertical')
    #plt.xlabel(feature_binned, fontsize=12)
    #plt.ylabel(target, fontsize=12)
    plt.show()
    
    
    

def replace_categories2(train_set,categorical_preds,num_categories):
    """
    Function that takes categorical variable, check its number of categories
    and if this number is higher than num_categories, categories will be merged together.
    """

    for i in categorical_preds:
        if train_set[i].nunique()>num_categories:
            print(i)
            print(train_set[i].nunique())
            top_n_cat=train_set[i].value_counts()[:10].index.tolist()
            train_set[i]=np.where(train_set[i].isin(top_n_cat),train_set[i],'other')   
    return train_set



def comp_auc(target, predictions):
    return roc_auc_score(target, predictions)


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


def one_model_rf(x_train, x_valid, y_train, y_valid):
    """Fit one RF model
    """

    # Firstly, we will try to find optimal numer of trees
    
    best_performance = 0
    opt_num_nn = 0
    
    for nn in [5,10,20,50,100]:
           
        forest = RandomForestClassifier(n_estimators=nn, max_depth=3, random_state=0)
        forest.fit(x_train, y_train)
        
        score = roc_auc_score(y_valid, forest.predict_proba(x_valid)[:,1])
        if score > best_performance:
            best_performance = score
            opt_num_nn = nn
            
    # Now we have optimal number of neigh
    
    forest = RandomForestClassifier(n_estimators = opt_num_nn, max_depth=3, random_state=0)
    forest.fit(x_train, y_train)

    return forest


def one_model_dt(x_train, x_valid, y_train, y_valid):
    """"Fit one DT model
    """

    # Firstly, we will try to find optimal numer of trees
    
    best_performance = 0
    opt_num_dpth = 0
    
    for depth in [2,3,4,5]:
           
        tree = DecisionTreeClassifier(random_state=0, max_depth=depth)
        tree.fit(x_train, y_train)
        
        score = roc_auc_score(y_valid, tree.predict_proba(x_valid)[:,1])
        if score > best_performance:
            best_performance = score
            opt_num_dpth = depth
            
    # Now we have optimal number of neigh
    
    tree = DecisionTreeClassifier(random_state=0, max_depth=opt_num_dpth)
    tree.fit(x_train, y_train)

    return tree



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




def train_model_CV(labels, data, preds,  target, cross_val, data_dummy, label):
    """
    Train cross-validation model
    """

    params={'early_stopping_rounds': 100,
        'num_boost_round': 1000,
        'learning_rate': 0.01,
        'metric': 'auc',
        'objective': 'binary',
        'max_depth': 2,
        'seed': 1234, 
        'verbosity': 0}

    
    train, test, y_train, y_test = train_test_split(data[preds], data[label], test_size=0.3, random_state = 99)
    train_dummy, test_dummy, _, _ = train_test_split(data_dummy, data[label], test_size=0.3, random_state = 99)
    #this will be done always same because of the random seed
    
    predictions_lgbm = np.zeros(test.shape[0])
    predictions_rf = np.zeros(test.shape[0])
    predictions_dt = np.zeros(test.shape[0])
    
    models_lgbm = []
    models_rf = []
    models_dt = []
    
    for train_indexes, valid_indexes in cross_val.split(train, y_train):
        train_data_lgbm= train.iloc[train_indexes]
        valid_data_lgbm = train.iloc[valid_indexes]
        
        train_data_rf= train_dummy.iloc[train_indexes]
        valid_data_rf = train_dummy.iloc[valid_indexes]
    
        train_target = y_train.iloc[train_indexes]
        valid_target = y_train.iloc[valid_indexes]
        
        
        models_lgbm.append(one_model_lgbm(train_data_lgbm, valid_data_lgbm, train_target, valid_target, params))
        models_rf.append(one_model_rf(train_data_rf, valid_data_rf, train_target, valid_target))
        models_dt.append(one_model_dt(train_data_rf, valid_data_rf, train_target, valid_target))
        
    for model in models_lgbm:
        predictions_lgbm = predictions_lgbm + model.predict(test[preds]) / len(models_lgbm)
        
    for model in models_rf:
        predictions_rf = predictions_rf + model.predict_proba(test_dummy)[:,1] / len(models_rf)
        
    for model in models_dt:
        predictions_dt = predictions_dt + model.predict_proba(test_dummy)[:,1] / len(models_dt)
    
    importance = plot_importance(models_lgbm, imp_type = 'Importance_gain', preds = preds ,ret=True, show=True, n_predictors = len(preds))
    
    return [roc_auc_score(y_test, predictions_lgbm),
           roc_auc_score(y_test, predictions_rf),
           roc_auc_score(y_test, predictions_dt),
           importance]

