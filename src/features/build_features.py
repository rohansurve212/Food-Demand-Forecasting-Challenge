# Import standard libraries
import sys
sys.path.append('C:/Users/rohan/Documents/Projects/Food_Demand_Forecasting_Challenge/Food_Demand_Forecasting_Challenge')

import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

def features_by_center(df):
    """
        Compute the mean values of Checkout Price (CP) and discount (D) for all meal_ids within a week--center_id combination    
    """
    
    df_ep = df.loc[df['emailer_for_promotion'] == 1, :]
    df_hf = df.loc[df['homepage_featured'] == 1, :]

    for data_set in [df, df_ep, df_hf]:
        
        for data_cut in [['week','center_id'],
                         ['week','center_id','category'],
                         ['week','center_id','cuisine']]:
    
            data_cut_name = "_".join(data_cut)
    
            for col in ['checkout_price','discount']:
            
                gdf = data_set.groupby(data_cut)[col].mean().reset_index()
            
                if data_set.equals(df_ep):
                    
                    # Set column names
                    gdf.columns = data_cut + [data_cut_name + '_ep_mean_' + col]
                
                    # Merge Mean with original dataset
                    df = pd.merge(df, gdf, on=data_cut, how='left')
                
                    # Compute difference between originl value and Mean
                    df[col + '_minus_' + data_cut_name + '_ep_mean_' + col] = (df[col] - df[data_cut_name + '_ep_mean_' + col])
                
                    # Check if original value is greater than Mean
                    df[col + '_gt_' + data_cut_name + '_ep_mean_' + col] = (df[col] > df[data_cut_name + '_ep_mean_' + col]).astype(int)
                
                    # Drop the Mean 
                    df = df.drop([data_cut_name + '_ep_mean_' + col], axis=1)
                
                elif data_set.equals(df_hf):
                
                    # Set column names
                    gdf.columns = data_cut + [data_cut_name + '_hf_mean_' + col]
                
                    # Merge Mean with original dataset
                    df = pd.merge(df, gdf, on=data_cut, how='left')
                
                    # Compute difference between originl value and Mean
                    df[col + '_minus_' + data_cut_name + '_hf_mean_' + col] = (df[col] - df[data_cut_name + '_hf_mean_' + col])
                
                    # Check if original value is greater than Mean
                    df[col + '_gt_' + data_cut_name + '_hf_mean_' + col] = (df[col] > df[data_cut_name + '_hf_mean_' + col]).astype(int)
                
                    # Drop the Mean 
                    df = df.drop([data_cut_name + '_hf_mean_' + col], axis=1)
                
                else:
                
                    # Set column names
                    gdf.columns = data_cut + [data_cut_name + '_mean_' + col]
                
                    # Merge Mean with original dataset
                    df = pd.merge(df, gdf, on=data_cut, how='left')
                
                    # Compute difference between originl value and Mean
                    df[col + '_minus_' + data_cut_name + '_mean_' + col] = (df[col] - df[data_cut_name + '_mean_' + col])
    
                    # Check if original value is greater than Mean
                    df[col + '_gt_' + data_cut_name + '_mean_' + col] = (df[col] > df[data_cut_name + '_mean_' + col]).astype(int)
    
                    # Drop the Mean 
                    df = df.drop([data_cut_name + '_mean_' + col], axis=1)
         
    return df


def total_meals_by_center(df):
    """
        Compute the total meals for each week--center_id combination    
    """
    # Compute the count of meal ids per week_center combination
    gdf = df.groupby(['week','center_id'])['meal_id'].count().reset_index()
    
    # Set column names
    gdf.columns = ['week','center_id','center-wise_meal_counts']
    
    # Merge count of meal ids with original data
    df = pd.merge(df, gdf, on=['week','center_id'], how='left')
    
    return df


# 

def features_by_ep_or_hf(df):
    """
        Find the number of meal_ids by category and cuisine that were featured on Homepage and number of meal_ids that were promoted by emailers
    """
    
    for data_cut in [['week','center_id','category'],
                     ['week','center_id','cuisine']]:
        
        data_cut_name = "_".join(data_cut)
        
        for col in ['homepage_featured','emailer_for_promotion']:
            
            # Compute sum of values for the data_cut cobination
            gdf = df.groupby(data_cut)[col].sum().reset_index()
            
            # Set column names
            gdf.columns = data_cut + [data_cut_name + '_sum_' + col]
            
            # Merge sum of values with original data
            df = pd.merge(df, gdf, on=data_cut, how='left')
            
    return df


def features_by_city_or_region(df):
    """
        Compute total and mean operating area for each region and city and ratio of center op area to total region op area and city op area
    """
    op_area_df = df[['center_id','region_code','city_code','op_area']].copy().drop_duplicates()
    
    for col in ['region_code','city_code']:
        
        # Compute total op_area for each col
        gdf_1 = op_area_df.groupby([col])['op_area'].sum().reset_index()
        
        # Set column names
        gdf_1.columns = [col, col + '_op_area']
        
        # Compute mean op_area for each col
        gdf_2 = op_area_df.groupby([col])['op_area'].mean().reset_index()
        
        # Set column names
        gdf_2.columns = [col, col + '_mean_op_area']
        
        # Merge total op_area with original data
        df = pd.merge(df, gdf_1, on=[col], how='left')
        
        # Compute ratio of op_area to total op_area 
        df['center_op_area_ratio_' + col + '_op_area'] = df['op_area'] / df[col +'_op_area']
        
        # Merge mean op_area with original data 
        df = pd.merge(df, gdf_2, on=[col], how='left')
        
        # Compute ratio of op_area to mean op_area
        df['center_op_area_gt_' + col + '_mean_op_area'] = (df['op_area'] > df[col + '_mean_op_area']).astype(int)
    
        # Drop total op_area and mean op_area columns
        df = df.drop([col +'_op_area', col + '_mean_op_area'], axis=1)
        
    return df


def temporal_features_set_1(df):
    """
        Check if a meal--center combination was promoted by email or featured on homepage last week 
        or the week before and the cumulative sum of all previous promotions and features
    """
    
    for col in ['emailer_for_promotion','homepage_featured']:
        
        # Compute value one week ago
        df[col+'_one_week_ago'] = df.groupby(['center_id','meal_id'])[col].shift(1).values
        
        # Compute value two weeks ago
        df[col+'_two_weeks_ago'] = df.groupby(['center_id','meal_id'])[col].shift(2).values
        
        # Compute cumulative value for all the previous weeks
        df[col+'_cum_sum'] = df.groupby(['center_id','meal_id'])[col].cumsum().values

    return df


def temporal_features_set_2(df):
    """
        Compute last week checkout price and last week discount of each meal--center combination  
        and check if current set of checkout price and discount is greater than last week's
    """
    
    for col in ['checkout_price','discount']:
        df = df.reset_index()
        
        # Compute previous week value
        df[col + '_prev_week'] = df.groupby(['center_id','meal_id'])[col].shift(1).values
        
        # Check if current week value is greater than previous week value
        df[col + '_gt_' + col + '_prev_week'] = (df[col] > df[col + '_prev_week']).astype(int)
        
        # Drop previous week value
        df = df.drop([col + '_prev_week'], axis=1)
        
    return df


def features_by_cui_or_cat(df):
    """
        Create Label Encoder features for different set of cuisine and categories
    """

    df['cui_ita_cat_bev_or_san'] = np.where((df['cuisine'] == 'Italian') & (df['category'].isin(['Beverages','Sandwich'])), 1, 0)
    df['cui_tha_cat_bev'] = np.where((df['cuisine'] == 'Thai') & (df['category'].isin(['Beverages'])), 1, 0)
    df['cui_ind_cat_ric'] = np.where((df['cuisine'] == 'Indian') & (df['category'].isin(['Rice Bowl'])), 1, 0)
    df['cui_con_cat_bev_or_piz'] = np.where((df['cuisine'] == 'Continental') & (df['category'].isin(['Beverages','Pizza'])), 1, 0)
    df['cui_tha_or_ita_cat_bev'] = np.where((df['cuisine'].isin(['Thai','Italian'])) & (df['category'] == 'Beverages'), 1, 0)
    
    return df