"""
Description: A Python utility contains different methods to 
create / work / update csv files
"""

import pandas as pd
import os
import re


# create table by month
def csv_from_parquet_per_month(dir_main, dir_to_save, columns, mark):
    '''dir_main - path to parquet files
    dir_to_save - path to store min tables
    columns - columns that shoul be extact from parquet files'''

    regex = '^[a-z].*'
    # create directories
    if not os.path.exists(f'{dir_to_save}'): 
        # if the folder doen't exist then create it.  
        os.makedirs(f'{dir_to_save}') 
        print(f'{dir_to_save}', 'was created')

    # This loop creates csv tables with specified colums data for each month
    for y in os.listdir(dir_main): # loop over folders
        if re.match(regex, y): # get only folders with data (year) 
            for m in os.listdir(f'{dir_main}/{y}'): # loop over folders
                if re.match(regex, m): # get only folders with data (month) 
                    df_min = None
                    for d in os.listdir(f'{dir_main}/{y}/{m}'): # loop over folders
                        if re.match(regex, d): # get only folders with data (day) 
                            tdf = pd.read_parquet(f'{dir_main}/{y}/{m}/{d}', engine='pyarrow')
                            tdf = tdf[columns]

                            # create small df
                            if df_min is not None:
                                df_min = pd.concat([df_min, tdf])
                            else:
                                df_min = tdf
                            
                    df_min.to_csv(f'{dir_to_save}/min_{mark}_df_{y[-4:]}_{m[-2:]}.csv')


# create one table from montly tables
def joint_csv(dir_w_month_csv, dir_to_save, mark):
    # create df_2 with all lim data

    regex = '^[a-z].*'
    joint_df = None

    for f in os.listdir(dir_w_month_csv):
        if re.match(regex, f):
            tdf = pd.read_csv(f'{dir_w_month_csv}/{f}', index_col=0)

            if joint_df is None:
                joint_df = tdf
            else: 
                joint_df = pd.concat([joint_df, tdf], ignore_index=True)

    # save to csv df_2
    joint_df.to_csv(f'{dir_to_save}/df_{mark}.csv', index=False)