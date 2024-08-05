"""
Description: A Python utility contains methods to 
create train and test datasets
"""

import pandas as pd
from itertools import combinations
from sklearn.model_selection import train_test_split

def get_last_user_sessions(df, columns=['session_id']):
    '''df - original df
    columns - columns that should be returned:
    - for the preprocessed ds with unique session_id it is only session_id
    - original data has ambiguious values in session_id: different users
    could have similar session_id. In that case unique identifiers are session_id + user_id'''


    idx = df.groupby('user_id')['session_timestamp'].idxmax()

    res = df.loc[idx, columns].reset_index(drop=True)

    if len(columns) == 1:
        return (list(res[columns[0]]))
    
    else:
        return res
    

def train_test_by_session_tmstp(df, columns=['session_id']):
    '''this function create the test dataset from the last sessions 
    of each user in df'''

    last_sessions = get_last_user_sessions(df, columns=columns)

    if len(columns)==1:
        train_set = df[~df['session_id'].isin(last_sessions)]

        test_set = df[df['session_id'].isin(last_sessions)]

    else:
        # in construction :)
        pass
    
    return train_set, test_set


def train_test_by_users(df, user_column='user_id', random_state = 42, test_size = 0.3):
    '''This function devides df by user_id'''

    train_user_set, test_user_set = train_test_split(df[user_column].unique(), test_size=test_size, random_state=random_state)

    train_set = df[df[user_column].isin(train_user_set)]
    test_set = df[df[user_column].isin(test_user_set)]
    
    return train_set, test_set

def train_val_test_split(df, user_column='user_id', unique_sess_columns=['session_id'], random_state = 42, test_size = 0.3):
    '''This function combines train_test_by_users and train_test_by_session_tmstp
    functions and returns train, val and test sets with the next rules:
    train and test sets contain sessions from different users,
    val set contains last session of each user in the train set
    
    Return: train_set, val_set, test_set'''

    tmp_train_set, test_set = train_test_by_users(df, user_column = user_column, random_state=random_state, test_size=test_size)

    train_set, val_set = train_test_by_session_tmstp(tmp_train_set, columns=unique_sess_columns)

    return train_set, val_set, test_set