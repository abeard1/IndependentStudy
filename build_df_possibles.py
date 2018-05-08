import pandas as pd
import networkx as nx
import numpy as np
import time


def build_year(year):
    file_read = 'pruned_data/pruned_data_{}.csv'.format(year)
    df_actual = pd.read_csv(file_read)

    # reformat so triangle generator can differentiate
    df_actual['defense'] = 'Defense_' + df_actual['opp_id']

    #change this so its the same as toy example
    df_actual.columns = ['player' if x == 'player_id' else x for x in df_actual.columns]

    # add per min stats
    stats = ['pts', 'trb', 'ast', 'blk', 'stl']

    for stat in stats:
        new_col = stat + '_per_min'
        df_actual[new_col] = df_actual.apply(lambda row: 0 if row['mp'] == 0 else row[stat] / row['mp'], axis=1)

    # build df_possible

    # initial player-defense pairs

    defense_list = []
    player_list = []

    for defense in df_actual['defense'].unique():
        for player in df_actual['player'].unique():
            defense_list.append(defense)
            player_list.append(player)
            
    data_dict = {'defense' : defense_list, 'player' : player_list}        

    df_possible = pd.DataFrame(data=data_dict)

    # real results from df_actual -> df_possible

    # apply above tests to all rows

    stats_per_min = ['pts_per_min', 'trb_per_min', 'ast_per_min', 'blk_per_min', 'stl_per_min']

    print('{} num of rows: {}'.format(year, len(df_possible)))    

    for index, row in df_possible.iterrows():
        if index % 1000 == 0:
            print('index: {}'.format(index))
            print(time.time())
        
        df_filtered = df_actual[(df_actual['player'] == row['player']) & (df_actual['defense'] == row['defense'])] 
        
        df_possible.loc[index, 'times_played'] = len(df_filtered)
        
        if len(df_filtered) == 0:
            for stat in stats_per_min:
                df_possible.loc[index, stat] = 0.0
        else:
            for stat in stats_per_min:
                df_possible.loc[index, stat] = df_filtered[stat].mean()

    # save csv
    file_name = 'df_possibles/df_possible_{}.csv'.format(year)
    player_df.to_csv(file_name, index=False)


years = [2015, 2016, 2017]

for year in years:
    build_year(year)


