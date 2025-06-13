#import packages
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_percentage_error as MAPE
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np


class RedemptionModel:

    def __init__(self, X, target_col):
        '''
        Args:
        X (pandas.DataFrame): Dataset of predictors, output from load_data()
        target_col (str): column name for target variable
        '''
        self._predictions = {}
        self.X = X
        self.target_col = target_col
        self.results = {} # model results
        self.predictions_base = {} # predictions used for scoring for base model
        self.predictions_improved = {} #predictions used for scoring for improved model
    
    #rather than using the MAPE score, I will simply look at the MSE for easier interpretability
    def score(self, truth, preds):
        return pd.DataFrame({
            'truth': truth,
            'prediction': preds,
            'error': truth - preds,
            'abs_error': (truth - preds).abs(),
            'squared_error': (truth - preds) ** 2
                }, index=truth.index)
    
    #scoring for improved model
    def score_improved(self, truth, preds_improved):
            return pd.DataFrame({
                'truth': truth,
                'prediction': preds_improved,
                'error': truth - preds_improved,
                'abs_error': (truth - preds_improved).abs(),
                'squared_error': (truth - preds_improved) ** 2
                    }, index=truth.index)

    #runs base and improved models
    def run_models(self, n_splits=4, test_size=365):
        '''Run the models and store results for cross validated splits in
        self.results.
        '''
        # Time series split
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
        cnt = 0 # keep track of splits
        for train, test in tscv.split(self.X):
            X_train = self.X.iloc[train]
            X_test = self.X.iloc[test]
            # Base model - please leave this here
            preds = self._base_model(X_train, X_test)
            if 'Base' not in self.results:
                self.results['Base'] = {}
            df_preds = self.score(X_test[self.target_col], preds)
            self.results['Base'][cnt] = self.score(X_test[self.target_col],
                                preds)
            
            #to be exported later
            self.predictions_base[cnt] = df_preds
            self.plot(preds, 'Base')
            
            # Improved Model
            preds_improved = self._base_model_improved(X_train, X_test)
            if 'Improved' not in self.results:
                self.results['Improved'] = {}
            df_preds_improved = self.score_improved(X_test[self.target_col], preds_improved)
            self.results['Improved'][cnt] = self.score_improved(X_test[self.target_col],
                                preds_improved)
            
            #to be exported later
            self.predictions_improved[cnt] = df_preds_improved
            self.plot_improved(preds_improved, 'Improved')
        
            cnt += 1
   

    def _base_model(self, train, test):
        '''
        Our base, too-simple model.
        Your model needs to take the training and test datasets (dataframes)
        and output a prediction based on the test data.

        Please leave this method as-is.

        '''
        res = sm.tsa.seasonal_decompose(train[self.target_col],
                                        period=365)
        res_clip = res.seasonal.apply(lambda x: max(0,x))
        res_clip.index = res_clip.index.dayofyear
        res_clip = res_clip.groupby(res_clip.index).mean()
        res_dict = res_clip.to_dict()
        return pd.Series(index = test.index, 
                         data = map(lambda x: res_dict[x], test.index.dayofyear))

    #improved model with weighting component
    def _base_model_improved(self, train, test):
        
        res = sm.tsa.seasonal_decompose(train[self.target_col],
                                        period=365)
        res_clip = res.seasonal.apply(lambda x: max(0,x))
       
        #adding index by year, to be used to weight observations
        res_clip_df = pd.DataFrame({'seasonal': res_clip})
        res_clip_df['dayofyear'] = res_clip_df.index.dayofyear
        res_clip_df['year'] = res_clip_df.index.year
        
        #weighting recent years more heavily
        unique_years = sorted(res_clip_df['year'].unique())
        year_weights = {year: i + 1 for i, year in enumerate(unique_years)}
        res_clip_df['weight'] = res_clip_df['year'].map(year_weights)

        weighted = res_clip_df.groupby('dayofyear').apply(
            lambda g: np.average(g['seasonal'], weights=g['weight'])
            )

        res_dict_improved = weighted.to_dict()
        return pd.Series(index=test.index, 
                 data=map(lambda x: res_dict_improved[x], test.index.dayofyear))

    #plots prediction vs. truth
    def plot(self, preds, label):
        # plot out the forecasts
        fig, ax = plt.subplots(figsize=(15, 5))
        ax.scatter(self.X.index, self.X[self.target_col], s=0.4, color='grey',
            label='Observed')
        ax.plot(preds, label = label, color='red')
        plt.legend()
    
    def plot_improved(self, preds_improved, label):
        # plot out the forecasts
        fig, ax = plt.subplots(figsize=(15, 5))
        ax.scatter(self.X.index, self.X[self.target_col], s=0.4, color='grey',
            label='Observed')
        ax.plot(preds_improved, label = label, color='red')
        plt.legend()
        
    #export results
    def export_results(self, filename='C:/Users/BillW/OneDrive/Documents/Apps/Data Tasks/Manitoba/base_model_results.csv'):
        all_dfs = []
        for fold, df in self.predictions_base.items():
            df_copy = df.copy()
            df_copy['fold'] = fold
            all_dfs.append(df_copy)
        full_df = pd.concat(all_dfs)
        full_df.to_csv(filename)
        
    def export_results_improved(self, filename='C:/Users/BillW/OneDrive/Documents/Apps/Data Tasks/Manitoba/improved_model_results.csv'):
        all_dfs = []
        for fold, df in self.predictions_improved.items():
            df_copy = df.copy()
            df_copy['fold'] = fold
            all_dfs.append(df_copy)
        full_df = pd.concat(all_dfs)
        full_df.to_csv(filename)


#creating new class for new model for simplicity
class SalesModel:

    def __init__(self, X, target_col):
        self._predictions = {}
        self.X = X
        self.target_col = target_col
        self.results = {} # model results
        self.predictions = {} #predictions used for scoring
    
    #scoring using same method as before
    def score_new(self, truth, preds_new):
        return pd.DataFrame({
            'truth': truth,
            'prediction': preds_new,
            'error': truth - preds_new,
            'abs_error': (truth - preds_new).abs(),
            'squared_error': (truth - preds_new) ** 2
                }, index=truth.index)
        

    #run new model
    def run_models(self, n_splits=4, test_size=365):
        '''Run the models and store results for cross validated splits in
        self.results.
        '''
        # Time series split
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
        cnt = 0 # keep track of splits
        for train, test in tscv.split(self.X):
            X_train = self.X.iloc[train]
            X_test = self.X.iloc[test]
            #New Model:
            preds_new = self._my_new_model(X_train, X_test)
            if 'New' not in self.results:
                self.results['New'] = {}
            df_preds_new = self.score_new(X_test[self.target_col], preds_new)
            self.results['New'][cnt] = df_preds_new
            
            #to be exported later
            self.predictions[cnt] = df_preds_new
            
            self.plot_new(preds_new, 'New')
        
            cnt += 1
        
    #new model with added variables
    def _my_new_model(self, train, test):
        
        #our predictor variables
        predictors = ['Mean Temp', 'Weekend']

        #Define variables in training and testing sets
        X_train = train[predictors]
        y_train = train[self.target_col]
        X_test = test[predictors]

        # Fit linear regression
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predict on test
        y_pred = model.predict(X_test)

        # Return predictions as a Series indexed like test
        return pd.Series(index=test.index, data=y_pred)
    
    #plots prediction vs. truth
    def plot_new(self, preds_new, label):
        # plot out the forecasts
        fig, ax = plt.subplots(figsize=(15, 5))
        ax.scatter(self.X.index, self.X[self.target_col], s=0.4, color='grey',
            label='Observed')
        ax.plot(preds_new, label = label, color='red')
        plt.legend()
     
    def export_results_new(self, filename='C:/Users/BillW/OneDrive/Documents/Apps/Data Tasks/Manitoba/new_model_results.csv'):
        all_dfs = []
        for fold, df in self.predictions.items():
            df_copy = df.copy()
            df_copy['fold'] = fold
            all_dfs.append(df_copy)
        full_df = pd.concat(all_dfs)
        full_df.to_csv(filename)
    

#run load data
#for base and improved models
def load_data(file):
    # generic data processing function
    df = pd.read_csv(file,
                dtype={'_id':int, 'Redemption Count': int, 'Sales Count':int},
                parse_dates=['Timestamp'])
    df = df.dropna(subset=['Redemption Count'])
    df.sort_values('Timestamp', inplace=True)
    # convert to time-aware
    df.set_index('Timestamp', inplace=True)
    df_daily = df.resample('d').sum()
    # engineer some time features 
    df_daily['monthly'] = df_daily.reset_index().Timestamp.dt.month.values
    df_daily['quarter'] = df_daily.reset_index().Timestamp.dt.quarter.values
    return df_daily
df = load_data('C:/Users/BillW/OneDrive/Documents/Apps/Data Tasks/Manitoba/Toronto Island Ferry Ticket Counts.csv')

#for brand new model
def load_data_new(file):
    # generic data processing function
    df_new = pd.read_csv(file,
                dtype={'_id':int, 'Redemption Count': int, 'Sales Count':int, 'Mean Temp':float, 'Weekday':float, 'Weekend':float},
                parse_dates=['Timestamp'])
    df_new.sort_values('Timestamp', inplace=True)
    # convert to time-aware
    df_new.set_index('Timestamp', inplace=True)
    df_new_daily = df_new.resample('d').sum()
    # engineer some time features 
    df_new_daily['monthly'] = df_new_daily.reset_index().Timestamp.dt.month.values
    df_new_daily['quarter'] = df_new_daily.reset_index().Timestamp.dt.quarter.values
    return df_new_daily
df_new = load_data_new('C:/Users/BillW/OneDrive/Documents/Apps/Data Tasks/Manitoba/Toronto Island Ferry Ticket Counts_new.csv')


#run models
#base and improved models
rm = RedemptionModel(df, 'Redemption Count')
rm.run_models()
rm.export_results()
rm.export_results_improved()

#brand new model
rmn = SalesModel(df_new, 'Sales Count')
rmn.run_models()
rmn.export_results_new()

# print model summary stats
rm.results
rmn.results


