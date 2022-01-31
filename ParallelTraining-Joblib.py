#!/usr/bin/env python
# coding: utf-8

# In[37]:


import pandas as pd
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
import numpy as np
from fbprophet import Prophet
from functools import partial
import time
from multiprocessing import cpu_count
from joblib import Parallel, delayed, parallel_backend
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings("ignore")
from hyperopt import space_eval


# In[30]:


class HelperFunctions():
    def getStores(self, data):
        grouped_by_store = data.groupby('store_id')

        stores = {}
        for key, item in grouped_by_store:
            stores[key] = grouped_by_store.get_group(key)

        return stores
    
    def train_test_split(self, df, year=2016):
        train, test = df[df['year'] < year], df[df['year'] >= year]
        return train, test


# In[31]:


class Evaluator():        
    def __init__(self, labelCol, predictionCol):
        self.labelCol = labelCol
        self.predictionCol = predictionCol
    
    def Mape(self, data):        
        return np.mean(abs((data[self.labelCol] - data[self.predictionCol]) / data[self.labelCol]))


# In[66]:


class ProphetModel():
    space = {
        'seasonality_mode':hp.choice('seasonality_mode',['multiplicative','additive']),
        'changepoint_prior_scale':hp.choice('changepoint_prior_scale',np.arange(.1,2,.1)),
        'holidays_prior_scale': hp.choice('holidays_prior_scale',np.arange(.1,2,.1)),
        'n_changepoints' : hp.choice('n_changepoints',np.arange(20,200,20)),
        'weekly_seasonality' : hp.choice('weekly_seasonality', [True, False]),
        'daily_seasonality' : hp.choice('daily_seasonality', [True, False]),
        'yearly_seasonality' : hp.choice('yearly_seasonality', [True, False])
    }
        
    def train(self, train, validation, params):
        if params is None:
            prophet = Prophet()
        else:
            prophet = Prophet(**params)

        prophetModel = prophet.fit(train, iter=3000)

        validation_result = validation[["store_id", "year", "month", "y"]]
        validation_result = validation_result.reset_index(drop=True)
        validation_result["yhat"] = prophetModel.predict(validation[['ds']])[["yhat"]]
        
        evaluator = Evaluator(labelCol="y", predictionCol="yhat")
        score = evaluator.Mape(validation_result)    
        
        print('score: {0} model: {1}'.format(score, 'Prophet'))
        return {'loss': score, 'status': STATUS_OK}
    
    def fit(self, data, labelCol):
        df = data.copy()
        df = df.rename(columns={labelCol: 'y'})    
        data = HelperFunctions()
        train, validation = data.train_test_split(df, 2015)
        train = train[['ds','y']]

        trials = Trials()
        best = fmin(partial(self.train, train, validation),
                    space=ProphetModel.space,
                    algo=tpe.suggest,
                    max_evals=5,
                    trials=trials)
        
        bestParams = space_eval(self.space, best)
        bestLoss = trials.best_trial['result']['loss']        
                
        prophetModel = Prophet(**bestParams)
        df = df[['ds','y']]
        prophetModel = prophetModel.fit(df)

        return bestLoss, bestParams, prophetModel


# In[67]:


class SarimaxModel():
    
    def __init__(self):
        self.p_values = np.arange(0, 2)
        self.d_values = np.arange(1, 2)
        self.q_values = np.arange(1, 4)
        self.P_values = np.arange(0, 2)
        self.D_values = np.arange(1, 2)
        self.Q_values = np.arange(0, 3)
        self.m_values = np.arange(7, 8)     
        
    def train(self, train, validation, arima_order, seasonalOrder):    
        try:          
            y_hat = validation.copy() 
            model = SARIMAX(train['sales'], order=arima_order, seasonal_order=seasonalOrder)        
            model_fit = model.fit()
            predict = model_fit.predict("2015-01-01", "2015-12-01", dynamic=True)
            y_hat['model_prediction'] = predict      

            evaluator = Evaluator(labelCol="sales", predictionCol="model_prediction")
            error = evaluator.Mape(y_hat) 
            
            #error = Mape(validation['sales'], y_hat.model_prediction)            
                        
            print('score: {0} model: {1}'.format(error, 'Sarimax'))
            return error, arima_order, seasonalOrder
        
        except Exception as e:                
            print(f"##### Skipped modelling with: {arima_order}, {seasonalOrder}\n")
            print(e)
            return -1, arima_order, seasonalOrder

    
    def evaluate(self, train, validation, p_values, d_values, q_values, P_values,
                 D_values, Q_values, m_values, parallel=True):    
        
        executor = Parallel(n_jobs=cpu_count()) 
        score = []

        if parallel==False:
            for p in p_values:
                for d in d_values:
                    for q in q_values:
                        for P in P_values:
                            for D in D_values:
                                for Q in Q_values:
                                    for m in m_values:        
                                        results = self.train(train, validation, (p,d,q), (P,D,Q,m)) 
                                        #print(results)
                                        score.append(results)                                        

        else:
            try:
                tasks = (delayed(self.train)(train, validation, (p,d,q), (P,D,Q,m)) for m in m_values for Q in Q_values for D in D_values for P in P_values for q in q_values for d in d_values for p in p_values)        
                results = executor(tasks)
                score.append(results)
            except Exception as e:
                print('Fatal Error....')
                print(e)

        return score
        
    def fit(self, data, labelCol):
        df = data.copy()
        data = HelperFunctions()
        train, validation = data.train_test_split(df, 2015)
        train = train[['ds',labelCol]]
        validation = validation[['ds',labelCol]]
        train.set_index('ds', inplace=True)
        validation.set_index('ds', inplace=True)
        train.index = pd.DatetimeIndex(train.index.values,
                                       freq=train.index.inferred_freq)
        validation.index = pd.DatetimeIndex(validation.index.values,
                                       freq=validation.index.inferred_freq)

        result = self.evaluate(train, validation, self.p_values, self.d_values, self.q_values,
                                         self.P_values, self.D_values, self.Q_values, self.m_values, False)
        
        scores=[]
        for tuple_list in result:            
            scores.append(tuple_list)

        scores.sort(key=lambda x: x[0])
        params = scores[0]
        
        df = df[['ds',labelCol]]
        df.set_index('ds', inplace=True)
        df.index = pd.DatetimeIndex(df.index.values,
                                       freq=df.index.inferred_freq)
        
        sarimaxModel = SARIMAX(df['sales'], order=params[1], seasonal_order=params[2])        
        sarimaxModel = sarimaxModel.fit()
        
        return params[0], (params[1], params[2]), sarimaxModel


# In[68]:


class ModelSelector():
    def getModel(self, data):
        models = {}
        for key, value in data.items():
                                    
            print("Store: ", key)
            result = Parallel(n_jobs=cpu_count(), prefer="threads")(delayed(self.parallelTraining)(model, value) 
                                                                     for model in ['prophet', 'sarimax'])                     
            
            bestLossProphet, bestParamsProphet, modelProphet = result[0]
            bestLossSarimax, bestParamsSarimax, modelSarimax = result[1]
            
            print(bestLossProphet, bestParamsProphet, modelProphet)
            print(bestLossSarimax, bestParamsSarimax, modelSarimax)            
            
            print("Best Loss Prophet: {0}".format(bestLossProphet))
            print("Best Loss Sarimax: {0}".format(bestLossSarimax))
            
            if bestLossProphet < bestLossSarimax:
                models[key] = ['Prophet', modelProphet]
            else:
                models[key] = ['Sarimax', modelSarimax]
            
        return models    
    
    def parallelTraining(self, model, data):
        if model == "prophet":
            prophet = ProphetModel()
            bestLossProphet, bestParamsProphet, prophetModel = prophet.fit(data, "sales")            
            return bestLossProphet, bestParamsProphet, prophetModel
        
        elif model == "sarimax":
            sarimax = SarimaxModel()
            bestLossSarimax, bestParamsSarimax, sarimaxModel = sarimax.fit(data, "sales")            
            return bestLossSarimax, bestParamsSarimax, sarimaxModel        


# In[69]:


class Driver():
    def main(self):
        train = pd.read_csv('train.csv', index_col = 0)
        test = pd.read_csv('test.csv', index_col = 0)
        
        train['ds'] = pd.to_datetime(train[['year', 'month']].assign(day=1))
        test['ds'] = pd.to_datetime(test[['year', 'month']].assign(day=1))
        
        helper = HelperFunctions()
        trainStores = helper.getStores(train)
        testStores = helper.getStores(test)        
        
        print("number of stores: {0}".format(len(trainStores)))                
        
        modelSelector = ModelSelector()
        return modelSelector.getModel(trainStores)    


# In[ ]:


if __name__ == "__main__":
    driver = Driver()
    models = driver.main()


# In[ ]:




