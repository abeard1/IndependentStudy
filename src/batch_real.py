import pandas as pd
import networkx as nx
import numpy as np
import time
import operator

import pickle

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

import fitting

# will add more
models = [LinearRegression(), RandomForestRegressor()]


import random
test_fraction = 0.05

years = [2015, 2016, 2017]
stats = ['pts_per_min', 'trb_per_min', 'ast_per_min', 'blk_per_min', 'stl_per_min']

results_dict = {}


for year in years:
    df_full = pd.read_csv('df_actuals/actual_df_{}.csv'.format(year))
    
    # split df
    indices = list(range(len(df_full)))
    num_hidden = int(test_fraction * len(df_full))
    hidden_indices = random.sample(indices, num_hidden)
    
    df_full['hidden'] = 'False'
    for index in hidden_indices:
        df_full.at[index, 'hidden'] = 'True'
        
    df_train = df_full[df_full['hidden'] == 'False']
    df_test = df_full[df_full['hidden'] == 'True']
    
    #print(df_full.index.values.tolist())
    #print(df_train.index.values.tolist())
    #print(df_test.index.values.tolist())




    
        
    # create graphs from respective datasets
    edge_attrs = stats + ['times_played']
    
    G_train = nx.from_pandas_edgelist(df_train, 'defense', 'player', edge_attrs).to_undirected()
    G_full = nx.from_pandas_edgelist(df_full, 'defense', 'player', edge_attrs).to_undirected()

    
    for stat in stats:    
        # same triangle_lists used for all strategies/models

        print('GENERATING TRIANGLE LISTS / Y FULL LISTS')
        print(year)
        print(stat)


        print('TRAIN')
        print('num of rows: {}'.format(len(df_train)))

        train_triangle_lists = []
        train_y_full_list = []

        count = 0
        for index, row in df_train.iterrows():
            if count % 500 == 0:
                print(count)
                print(time.time())
            count += 1

            tl = strategies.generate_triangles(G_train, row['defense'], row['player'], stat)
            train_triangle_lists.append(tl)

            train_y_full_list.append(row[stat])

        print('TEST')
        print('num of rows: {}'.format(len(df_test)))

        test_triangle_lists = []
        test_y_full_list = []

        count = 0
        for index, row in df_test.iterrows():
            if count % 500 == 0:
                print(count)
                print(time.time())
            count += 1

            tl = strategies.generate_triangles(G_full, row['defense'], row['player'], stat)
            test_triangle_lists.append(tl)

            test_y_full_list.append(row[stat])


        for selection_method in strategies.selection_methods:
            # same matrices used for all models

            print('SELECTION METHOD')
            print(str(selection_method))

            
            print('BUILDING TRAIN X/Y MATRICES')
            print(time.time())

            train_x_list = []
            train_y_list = []

            for index in range(len(train_triangle_lists)):
                x = selection_method(train_triangle_lists[index])
                
                # only add x, y to lists if x actually returned value
                if type(x) == np.ndarray:
                    train_x_list.append(x)
                    y = np.array([train_y_full_list[index]])
                    train_y_list.append(y)

            # lists -> matrices
            train_x = np.vstack(train_x_list)
            train_y = np.vstack([arr.reshape((1, 1)) for arr in train_y_list])

            print(time.time())

            print('BUILDING TEST X/Y MATRICES')
            print(time.time())

            test_x_list = []
            test_y_list = []

            for index in range(len(test_triangle_lists)):
                x = selection_method(test_triangle_lists[index])
                
                # only add x, y to lists if x actually returned value
                if type(x) == np.ndarray:
                    test_x_list.append(x)
                    y = np.array([test_y_full_list[index]])
                    test_y_list.append(y)

            # lists -> matrices
            test_x = np.vstack(test_x_list)
            test_y = np.vstack([arr.reshape((1, 1)) for arr in test_y_list])

            print(time.time())
                
            
            for model in models:
                # train model and use to predict test_y

                print('FITTING MODEL/SAVING RESULTS')
                print(str(model))
                print(time.time())
            
                model.fit(train_x, train_y)
            
                print(time.time())
            
                test_y_pred = model.predict(test_x)
            
                # save results
                key = (year, stat, str(selection_method), str(model))
                results_dict[key] = (len(test_y_full_list), test_y, test_y_pred)

                with open('results2.pickle', 'wb') as handle:
                    pickle.dump(results_dict, handle)

