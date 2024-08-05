"""
Description: A Python utility contains methos to preprocessing data
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from collections import Counter
import ast

from ua_parser import user_agent_parser

from configuration.constants import COLUMNS_UNIQ_VAL as cuv
from configuration.constants import COLUMNS_AMB_VAL as cav


  
    
# Convert ID Columns to Numerical
class IDTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column, unique_index=None):

        # yet for all columns
        self.column = column

        # unique index is needed in cases with the new data
        self.unique_index = unique_index
        # return self

    def fit(self, X, y=None, save=True):
        upd_df = X.copy()

        # get data about the column from the constants file
        col = cuv.id_columns_dict[self.column]
        # self.map_df_dict = {}

        try:
            tmp_df = pd.reas_csv(f"../processed_data/df_{col[0][-1].replace('_id', 's')}{('_' + self.unique_index) if self.unique_index else ''}.csv")
        except:
            tmp_df = pd.DataFrame(upd_df[col[0]]).drop_duplicates().reset_index(drop=True).reset_index() # double reset to keep consistent order
            tmp_df = tmp_df.rename(columns={'index': f'{col[0][-1]}_new'})
            tmp_df[f'{col[0][-1]}_new'] = tmp_df[f'{col[0][-1]}_new'] + col[1]

            # save to csv for reversibility
            if save:
                tmp_df.to_csv(f"../processed_data/df_{col[0][-1].replace('_id', 's')}{('_' + self.unique_index) if self.unique_index else ''}.csv", index=False)

        self.map_df = [tmp_df, col[0]]

        return self
    
        
    def transform(self, X):
        upd_df = X.copy()

        # merge with the main df

        try:
            upd_df = upd_df.merge(self.map_df[0], on=self.map_df[1])
            
            upd_df[self.column] = upd_df[f'{self.column}_new']
            upd_df = upd_df.drop(columns=[f'{self.column}_new'])

        except:
            print(f"No columns {' '.join(self.map_df[1])}")

        return upd_df

    def get_feature_names_out(self, input_features=None):
        return [self.columns]
    

# Convert Timestamp Columns to Datetime
class TimestampTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column):
        self.column = column

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        X_copy[self.column] = pd.to_datetime(X_copy[self.column])
        return X_copy  

    def get_feature_names_out(self, input_features=None):
        return [self.column]

# Skip First Digits
class CustomStringTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column, skip_digits):
        self.column = column
        self.skip_digits = skip_digits

    def fit(self, X, y=None):
        X[self.column].fillna('', inplace=True)
        return self 
    
    def transform(self, X):
        X_copy = X.copy()
        X_copy[self.column] = X_copy[self.column].astype(str).apply(lambda x: x[self.skip_digits:] if isinstance(x, str) and len(x) >= self.skip_digits else x)
        return X_copy  

    def get_feature_names_out(self, input_features=None):
        return [self.column]


# extract informantion from session_timestamp
class HourWeekdayExtrTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column='session_timestamp'):
        self.column = column

    def fit(self, X, drop_col=True):
        self.drop_col = drop_col
        return self
    
    def transform(self, X):
        upd_df = X.copy()

        # convert to datetime is case if the column has other format
        upd_df[self.column] = pd.to_datetime(upd_df[self.column] , unit='ms')

        # extract session hour
        upd_df[f'{self.column.replace('timestamp', 'hour')}'] = upd_df[self.column].dt.hour

        # extract session day of the week 
        upd_df[f'{self.column.replace('timestamp', 'weekday')}'] = upd_df[self.column].dt.weekday

        if self.drop_col:
            upd_df = upd_df.drop(columns=[self.column])

        return upd_df

    def get_feature_names_out(self, input_features=None):
        return [self.column]
    
    
# extract informantion from user_agent
class UserAgentParserTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column='user_agent'):
        self.column = column

    def fit(self, X, drop_col=True):
        self.drop_col = drop_col
        return self
    
    def ua_parser(self, df):
        pua_df = df['user_agent'].apply(lambda x: user_agent_parser.ParseUserAgent(x)).apply(pd.Series)
        pua_df = pua_df.fillna('')
        df['user_agent_fam'] = pua_df['family']
        df['user_agent_fam_mjr'] = pua_df['family'] + '_' + pua_df['major']
        df['user_agent_mjr_mnr_ptch'] = pua_df['major'] + '_' + pua_df['minor'] + '_' + pua_df['patch']
        return df
    
    def os_parser(self, df):
        pos_df = df['user_agent'].apply(lambda x: user_agent_parser.ParseOS(x)).apply(pd.Series)
        pos_df = pos_df.fillna('')
        df['os_fam'] = pos_df['family']
        df['os_fam_mjr'] = pos_df['family'] + '_' + pos_df['major']
        df['os_mjr_mnr_ptch'] = pos_df['major'] + '_' + pos_df['minor'] + '_' + pos_df['patch']
        return df

    
    def transform(self, X):
        upd_df = X.copy()

        # ParseUserAgent
        upd_df = self.ua_parser(upd_df)

        # get device brand
        upd_df['device_brand'] = upd_df['user_agent'].apply(lambda x: user_agent_parser.ParseDevice(x)).apply(pd.Series)['brand']

        # ParseOS
        upd_df = self.os_parser(upd_df)

        if self.drop_col:
            upd_df = upd_df.drop(columns=[self.column])

        return upd_df

    def get_feature_names_out(self, input_features=None):
        return [self.column]
    

class IPParserTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column='ip_address'):
        self.column = column

    def fit(self, X, drop_col=True):
        self.drop_col = drop_col
        return self

    def transform(self, X):
        upd_df = X.copy()

        # Split the IP addresses into separate parts
        ip_parts = upd_df[self.column].str.split('.', expand=True)

        # Take only the first 4 columns to deal with potential outliers
        ip_parts = ip_parts.iloc[:, :4]
        ip_parts.columns = ['ip_address_1', 'ip_address_2', 'ip_address_3', 'ip_address_4']

        # Convert all entries to strings and check if they are digits, replace others with NaN
        ip_parts = ip_parts.map(lambda x: x if isinstance(x, str) and x.isdigit() else None)

        # Fill NaN values with 0
        ip_parts.fillna(value=0, inplace=True)

        # Convert columns to integers
        ip_parts = ip_parts.astype(int)

        # Join the new IP columns back to the original DataFrame
        upd_df = upd_df.join(ip_parts)

        # Optionally drop the original IP address column
        if self.drop_col:
            upd_df = upd_df.drop(columns=[self.column])

        return upd_df  

    def get_feature_names_out(self, input_features=None):
        return [f"{self.column}_{i}" for i in range(1, 5)]
    

class SizesTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column='sizes'):
        self.column = column

    def fit(self, X, drop_col=True):
        self.drop_col = drop_col
        return self
    
    def transform(self, X):
        upd_df = X.copy()

        # get the most common value in the entrie
        upd_df['most_common_size'] = upd_df[self.column].apply(self.find_most_common)

        # mapping
        mapping = {'14Y': 1, 'XS': 2, 'S': 3, 'M': 4, 'L': 5, 'XL': 6, 'XXL': 7, '3XL': 8}

        # Apply mapping
        upd_df['most_common_size_enc'] = upd_df['most_common_size'].map(mapping)

        # fill na
        upd_df['most_common_size_enc'] = upd_df['most_common_size_enc'].fillna(0)

        if self.drop_col:
            upd_df = upd_df.drop(columns=[self.column, 'most_common_size'])


        return upd_df

    def get_feature_names_out(self, input_features=None):
        return [self.column]
    
    
    def find_most_common(self, lst_str):
        try:
            lst = ast.literal_eval(lst_str)  # Safely evaluate string to a list
            if lst:  # Ensure the list is not empty
                return Counter(lst).most_common(1)[0][0]
        except (ValueError, SyntaxError):
            return None  # Return None if there's an error in conversion
        return None
    
class BoolToNumTransformer(BaseEstimator, TransformerMixin):
   
    def __init__(self, column):
        self.column = column
        # return self
    
    def fit(self, X):
        return self
    
    def transform(self, X, y=None):
        upd_df = X.copy()

        upd_df[self.column] = upd_df[self.column].astype(int)
                   
        return upd_df    
    

####   On a Session Level  ####

# Define custom transformer for aggregating session data
class AggregateCountTransformer(BaseEstimator, TransformerMixin):
   
    def __init__(self, column):
        self.column = column
        # return self
    
    def fit(self, X, drop_col=True):
        self.drop_col = drop_col
        return self
    
    def transform(self, X, y=None):
        upd_df = X.copy()

        upd_df[f'{self.column}_count'] = upd_df.groupby('session_id')[self.column].transform('count')

        if self.drop_col:
            upd_df = upd_df.drop(columns=[self.column])
                   
        return upd_df
    

# Define custom transformer for aggregating session data
class AggregateSumTransformer(BaseEstimator, TransformerMixin):
   
    def __init__(self, column):
        self.column = column
        # return self
    
    def fit(self, X, drop_col=True):
        self.drop_col = drop_col
        return self
    
    def transform(self, X, y=None):
        upd_df = X.copy()

        upd_df[f'{self.column}_sum'] = upd_df.groupby('session_id')[self.column].transform('sum')

        if self.drop_col:
            upd_df = upd_df.drop(columns=[self.column])
                   
        return upd_df
    

# calculate approximate length of session
class CalcLengthTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column='pageview_timestamp'):
        self.column = column

    def fit(self, X,  drop_col=True):
        self.drop_col = drop_col
        return self
    
    def transform(self, X, y=None):
        upd_df = X.copy()

        # Calculate max and min timestamps for each session
        grouped = upd_df.groupby('session_id')[self.column].agg(['min', 'max'])
        grouped = grouped.reset_index()

        # Calculate the session length by subtracting min from max, divide by 60000 to get min
        grouped['session_length_apx_min'] = round((grouped['max'] - grouped['min'])/60000, 2)
        grouped = grouped.drop(columns=['min', 'max'])

        # Merge this result back into the original DataFrame 
        upd_df = upd_df.merge(grouped, on='session_id', how='left')
        

        if self.drop_col:
            upd_df = upd_df.drop(columns=[self.column])

        return upd_df
    

# reducing ambiguity and grouping by session
class GroupBySessionTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass  

    def fit(self, X, y=None):
        return self  # Nothing to do here
    
    def safe_mode(self, s):
        modes = s.mode()
        if len(modes) > 0:
            return modes[0]
        return 0  # Returns 0 if no mode found
    
    def transform(self, X):
        upd_df = X.copy()

        # Fill with the most common value if there is ambiguity along a session
        agg_funcs = {col: self.safe_mode for col in upd_df.columns if col != 'session_id'}
        upd_df = upd_df.groupby("session_id").agg(agg_funcs).reset_index()
   
        return upd_df


class OneHotTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column, drop_col=True):
        self.column = column
        self.drop_col = drop_col

    def fit(self, X,  file_path='../data_pipeline/feature_categories_dict.pkl'):

        with open(file_path, 'rb') as f:
            self.feature_categories_dict = pickle.load(f)

        return self
    
    
    def transform(self, X, y=None):
        upd_df = X.copy()

        # Create a new column for each top category
        for category in self.feature_categories_dict[self.column]:
            upd_df[f"{self.column}_{category}"] = (upd_df[self.column] == category).astype(int)

        if self.drop_col:
            upd_df = upd_df.drop(columns=[self.column])

        return upd_df


class ScalerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scaler = StandardScaler()
    
    def fit(self, X, y=None):
        df_upd = X.copy()
        self.scaler.fit(df_upd.iloc[:, 2:])
        return self
    
    def transform(self, X):
        df_upd = X.copy()

        # apply for all columns except session_id and user_id
        df_upd.iloc[:, 2:] = self.scaler.transform(df_upd.iloc[:, 2:])
        return df_upd
    

class MinMaxScalerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scaler = MinMaxScaler()
    
    def fit(self, X, y=None):
        df_upd = X.copy()
        self.scaler.fit(df_upd.iloc[:, 2:])
        return self
    
    def transform(self, X):
        df_upd = X.copy()

        # apply for all columns except session_id and user_id
        df_upd.iloc[:, 2:] = self.scaler.transform(df_upd.iloc[:, 2:])
        return df_upd

