# Import Libraries
'''Standard libraries'''
import numpy as np
import pandas as pd
import sys, os, time
sys.path.append('C:/Users/rohan/Documents/Projects/Food_Demand_Forecasting_Challenge/Food_Demand_Forecasting_Challenge')

'''ML Algos'''
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.metrics import mean_squared_log_error

'''Misc'''
import itertools
from tqdm import tqdm
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")


def make_combinations(df, feat1, feat2):
    """
        Make all possible combinations (tuples) with one element from feat1 and other from feat2 and output it in the form of list
    """
    feat1_list = np.sort(df[feat1].unique())
    feat2_list = np.sort(df[feat2].unique())
    feat1_feat2_combo_list = []
    for r in itertools.product(feat1_list, feat2_list): 
        feat1_feat2_combo_list.append(r)

    return feat1_feat2_combo_list


def concatenate_features(df, feat1, feat2):
    """
        Make a dataframe of feat1, feat2 and the two concatenated into one string
    """
    feat1_feat2_combo_list = make_combinations(df, feat1, feat2)
    feat1_feat2_combo_df = pd.DataFrame(feat1_feat2_combo_list, columns=[feat1, feat2])
    feat1_feat2_combo_df[feat1+'_'+feat2] = feat1_feat2_combo_df[[feat1,feat2]].apply(lambda x: str(x[0])+'-'+str(x[1]), axis=1)

    return feat1_feat2_combo_df[feat1+'_'+feat2]


def make_full_series_combo(df, feat1, feat2):
    """
        Make a full series of concatenated feat1 and feat2 combo strings for the time period of the entire time series
    """
    time_series_period = len(df['week'].value_counts().index.to_list())

    feat1_feat2_combo_df[feat1+'_'+feat2] = concatenate_features(df, feat1, feat2)

    full_feat1_feat2_combo_id = pd.DataFrame()
    for combo in list(feat1_feat2_combo_df[feat1+'_'+feat2]):
        combo_df = pd.DataFrame(data=[combo]*time_series_period, columns=[feat1+'_'+feat2+'_combo'])
        full_feat1_feat2_combo_id = full_feat1_feat2_combo_id.append(combo_df, ignore_index=True)

    return full_feat1_feat2_combo_id



def make_time_series(df):
    """
        Prepare a dataframe in the time series format in order to be fed to Prophet time series model
    """
    # Calculate unique no. of meals and centers to compute list of all possible meal--center ids
    unique_number_of_meals   = len(np.sort(df['meal_id'].unique()))
    unique_number_of_centers = len(np.sort(df['center_id'].unique()))
    unique_combos = unique_number_of_meals * unique_number_of_centers
    time_series_period = len(df['week'].value_counts().index.to_list())

    df_time_series = df[['week','center_id','meal_id','num_orders']]

    # Sort the values on first center id, then meal id and then week number
    df_time_series = df_time_series.sort_values(by=['center_id','meal_id','week'])

    # Combine center id and meal id into a single string
    df_time_series['center_id_meal_id_combo'] = df_time_series[['center_id','meal_id']].apply(lambda x: str(x[0])+'-'+str(x[1]), axis=1)

    # Make a basic time series with just the week numbers 
    basic_raw_time_series = pd.DataFrame(data=np.sort(df_time_series['week'].value_counts().index.to_list()), columns=['week'])

    # Make a series of dates corresponding to the week numbers starting from 1st Jan 2000
    tdf = pd.date_range(start='2000-01-01', periods=time_series_period, freq='W').to_frame(index=False, name='ds')
    basic_raw_time_series = pd.concat([basic_raw_time_series,tdf], axis=1)

    # Make a full raw time series by repeating the basic raw time series over for all possible meal-center combos
    full_raw_time_series = pd.DataFrame()
    for i in range(unique_combos):
        full_raw_time_series = full_raw_time_series.append(basic_raw_time_series, ignore_index=True)

    # Make a full series of concatenated center_meal_combo strings for the time period of the entire time series
    full_center_meal_combo_id = make_full_series_combo(df_time_series, 'center_id', 'meal_id')

    # Concatenate full_raw_time_series with full_center_meal_combo_id
    full_raw_time_series = pd.concat([full_raw_time_series, full_center_meal_combo_id], axis=1)

    # Merge full_raw_time_series with df_time_series
    full_raw_time_series = full_raw_time_series.merge(df_time_series, how='left', on=['center_id_meal_id_combo','week'])

    # Sort the values on first center id, then meal id and then week number
    full_raw_time_series = full_raw_time_series.sort_values(by=['center_id_meal_id_combo','week'])

    # Format the date column to pandas datetime format
    full_raw_time_series['ds'] = pd.to_datetime(full_raw_time_series['ds'], format='%Y-%m-%d')

    # Rename the 'num_orders' column to 'y_actual' and impute Nan values with 0
    full_raw_time_series['y'] = full_raw_time_series['num_orders'].apply(lambda x: x if np.isnan(x)==False else 0)

    # Drop irrelevant columns from the data
    full_raw_time_series = full_raw_time_series.drop(['center_id','meal_id','num_orders'], axis=1, inplace=False)

    return full_raw_time_series


    


