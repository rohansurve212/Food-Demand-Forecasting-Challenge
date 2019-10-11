# Import Standard Libraries
import sys
sys.path.append('C:/Users/rohan/Documents/Projects/Food_Demand_Forecasting_Challenge/Food_Demand_Forecasting_Challenge')

import numpy as np
import pandas as pd

from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

import warnings
warnings.filterwarnings("ignore")


def drop_irrelevant_features(df):
    """
        Drop irrelevant columns from features set
        :param data_df: features_data
        :return: a dataframe of relevant features
    """
    cols_to_exclude = ['index', 'level_0', 'id']
    df = df.drop(cols_to_exclude, axis=1)

    return df


def separate_num_cat_features(df):
    """
        Takes a pandas Dataframe as input and
        separates numerical features from categorical features
        :param df_x: full_data
        :return: tuple of numerical features and categorical features
    """
    df[['city_code','region_code','center_id','meal_id']] = df[['city_code','region_code','center_id','meal_id']].astype(str)
    df_num_features = df.loc[:, df.select_dtypes(include=[np.number]).columns.tolist()]
    df_cat_features = df.loc[:, ['category','cuisine','center_type','city_code','region_code','center_id','meal_id']]

    return df_num_features, df_cat_features


def transform_num_features(df_num_features):
    """
        Takes a pandas Dataframe as input and transforms the numerical features (imputing missing values with median value and
        normalizing the values in each column such that minimum is 0 and maximum is 1)
        Arguments: data - DataFrame
        Returns: A Dataframe of transformed numerical features
    """

    # Let's build a pipeline to transform numerical features
    columns = df_num_features.columns.tolist()

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant')),
        ('normalizer', MinMaxScaler())
    ])

    df_num_tr = num_pipeline.fit_transform(df_num_features)
    df_num_tr = pd.DataFrame(df_num_tr, columns=columns)

    return df_num_tr


def transform_cat_features(df_cat_features):
    """
        Takes a pandas DataFrame as input and
        transforms the categorical features (imputing missing values with most frequent occurence
        value)
        Arguments: data - DataFrame
        Returns: A sparse matrix of transformed categorical features
    """

    # Let's build a pipeline to transform categorical features
    columns = df_cat_features.columns.tolist()
    imputer = SimpleImputer(strategy='most_frequent')
    df_cat_tr = imputer.fit_transform(df_cat_features)
    df_cat_tr = pd.DataFrame(df_cat_tr, columns=columns)

    return df_cat_tr


# def convert_spmatrix_to_dataframe(sparse_matrix):
#     """
#         Takes a sparse matrix and converts it to a dataframe
#         Arguments: A sparse matrix
#         Return: A Dataframe
#     """
#     columns = ['column_'+ str(num+1) for num in range(sparse_matrix.shape[1])]
#     df = pd.DataFrame(sparse_matrix.toarray(), columns=columns)
    
#     return df


def transform_all_features(df_num_features_transformed, df_cat_features_transformed):
    """
    transforms all the features to be ready to be fed to machine learning models
    :param data_df_x: data features
    :param data_df_num_feature_names: list of names of numerical features
    :param data_df_cat_feature_names: list of names of categorical features
    :return: A numpy array of transformed features
    """

    # Let's merge both numerical and categorical features into a single dataframe
    df_all_features_transformed = pd.concat([df_num_features_transformed, df_cat_features_transformed], axis=1)

    return df_all_features_transformed





