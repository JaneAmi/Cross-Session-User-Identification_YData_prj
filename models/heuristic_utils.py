"""
Description: A Python utility contains methods to 
implement heuristic approach, meaning finding
similas users based on external indentificators in 
a session
"""

import pandas as pd
import pickle
from tqdm import tqdm

from configuration.constants import COLUMNS_UNIQ_VAL as cuv

def create_ext_ids_dicts(df, save=True, file_path = 'external_ids_dicts.pkl'):
    '''The function create dictionaries for each external id
    where key is a external id value and value is a list of users
    that have sessions with this external id'''

    ext_ids_columns = cuv.EXT_IDS_COLUMNS
    res_dict = {}

    for col in ext_ids_columns:
        tmp_dict = df.groupby(col)['user_id'].unique().to_dict()

        # Convert each array in tmp_dict to a list
        tmp_dict = {k: list(v) for k, v in tmp_dict.items()}

        res_dict[col] = tmp_dict

    if save:
        with open(file_path, 'wb') as f:
            pickle.dump(res_dict, f)
    
    return res_dict


def upload_ext_ids_dicts(file_path = 'external_ids_dicts.pkl'):
    '''The function read the pickle file from file_path
    and upload dictionaries for each external id
    where key is a external id value and value is a list of users
    that have sessions with this external id'''

    with open(file_path, 'rb') as f:
        ext_ids_dicts = pickle.load(f)

    return ext_ids_dicts


def find_sim_users_in_ext_id_dict(session, ext_ids_dicts, verbose=False):
    '''This function creates a list of users that have sessions with 
    similar external ids as the input session
    and yields a dictionary where a key is an external id column
    and value is a list of users'''

    ext_ids_columns = cuv.EXT_IDS_COLUMNS

    ext_ids_columns_upd = ext_ids_columns.copy()

    # for each ext_ids_columns check that it doesn't empty within a session and update the list if needed
    for col in ext_ids_columns:
        if len(session[col].value_counts()) == 0:
            ext_ids_columns_upd.remove(col)

    res_dict = {}
    new_ext_ids_info = {}

    for col in ext_ids_columns:
        # check if the current session has this external id
        if col not in ext_ids_columns_upd:
            res_dict[col] = []
        
        else:
            try:
                sim_users = ext_ids_dicts[col][session[col].iloc[0]]
                res_dict[col] = list(sim_users)
                
            except:
                res_dict[col] = []
                # if the external id not in the original dict we add it into new_ext_ids_info
                # in case we want to update original dict later
                tmp_dict = {session[col].iloc[0]: session['user_id'].iloc[0]}
                new_ext_ids_info[col] = tmp_dict
    
    return res_dict, new_ext_ids_info
    

def create_results_of_heuristic_approach(val_df, file_path='external_ids_dicts.pkl'):
    '''This function go over each session in val_df
    and yield the df with the results of the function
    find_sim_users_in_ext_id_dict for each session'''

    # extract only relevant columns
    tmp_cols = ['user_id', 'session_id']
    tmp_cols.extend(cuv.EXT_IDS_COLUMNS)
    val_df_upd = val_df[tmp_cols].copy()

    unique_sessions = val_df_upd['session_id'].unique()
    ext_ids_dicts = upload_ext_ids_dicts(file_path)

    results = []

    for sss in tqdm(unique_sessions, desc='Processing sessions'):
        session = val_df_upd[val_df_upd['session_id']==sss]
        sess_res, new_ext_ids_info = find_sim_users_in_ext_id_dict(session, ext_ids_dicts)
        # we don't need new_ext_ids_info for the evaluation
        
        # add info about the session and original user_id
        sess_res['session_id'] = sss
        sess_res['or_user_id'] = session['user_id'].iloc[0]
        
        results.append(sess_res)

    res_df = pd.DataFrame(results)

    # update table with results
    # adding the bool columns that shows is original user_id is in user_ids returned
    # by the external id
    def check_user_id_s(x, col):
        return x['or_user_id'] in x[col]

    for col in cuv.EXT_IDS_COLUMNS:
        res_df[f"{col}_res"] = res_df.apply(lambda x: check_user_id_s(x, col), axis=1)

    return res_df


# functions for evaluation

# check precision
def precision_check(df):
    for col in cuv.EXT_IDS_COLUMNS:
        tmp_df = df[df[col].apply(lambda x: len(x) > 0)]
        res = tmp_df[f"{col}_res"].sum() / len(tmp_df)
        print(f"For {col} precision is: {res}")

# check accuracy
def accuaracy_check(df):
    for col in cuv.EXT_IDS_COLUMNS:
   
        res = df[f"{col}_res"].sum() / len(df)
        print(f"For {col} accuracy is: {res}")

# check confidence of the result, meaning how sure we can be if we take randomle one user_id from the result
def confidence_check(df):
    for col in cuv.EXT_IDS_COLUMNS:
        tmp_df = df[df[f"{col}_res"]>0]

        mean_length = tmp_df[col].apply(len).mean()
        res = 1 / mean_length
        print(f"For {col} probability that true result is correct is: {res}")


def evaluate_heuristic_results(df):
    print('Check precision')
    precision_check(df)
    print()
    print('Check accuracy')
    accuaracy_check(df)
    print()
    print('Check confidence')
    confidence_check(df)
