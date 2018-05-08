import pandas as pd
import networkx as nx
import numpy as np
import time
import operator

# holds different selection methods and triangle generator
import fitting

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


lin_reg = LinearRegression()
random_forest = RandomForestRegressor()

# RUN

import random
test_fraction = 0.05
num_iterations = 3

years = [2015, 2016, 2017]

stats = ['pts_per_min', 'trb_per_min', 'ast_per_min', 'blk_per_min', 'stl_per_min']


for year in years:
    df_full = pd.read_csv('df_actuals/df_actual_{}.csv'.format(year))
    

    for i in range(num_iterations):
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
            
            print('NEW RUN')
            print(year)
            print(stat)
            
            # do for all treatments at once
            
            # TRAIN
            
            train_x_lists = {}
            train_y_lists = {}
            
            train_x_actual = {}
            train_y_actual = {}
            
            for treatment in fitting.treatments:
                train_x_lists[treatment] = []
                train_y_lists[treatment] = []
                
            
            print('TRAIN')
            print('num of rows: {}'.format(len(df_train)))
                
            # some problem with indexes - not sure what yet
            count = 0
            for index, row in df_train.iterrows():
                if count % 500 == 0:
                    print(count)
                    print(time.time())
                count += 1
                
                # base arr is same for all treatments
                triangle_list = fitting.generate_triangles(G_train, row['defense'], row['player'], stat)
                
                # if <10 (because of diffs between G_full and G_train) -> leave out, test data will still be consistent
                if len(triangle_list) < 10:
                    continue
                
                triangle_arr = fitting.sort_triangles(triangle_list)
                
                for treatment in fitting.treatments:
                    x = fitting.route_treatment(triangle_arr, treatment)
                    
                    #if x == 'NAN':
                    #    print(row['defense'])
                    #    print(row['player'])
                    #    print('tlist with G_train: {}'.format(triangle_list))
                    #    withfull = fitting.generate_triangles(G_full, row['defense'], row['player'], stat)
                    #    print('tlist with G_full: {}'.format(withfull))
                
                    # only add x,y to list if x actually returned value
                    if type(x) == np.ndarray:
                        train_x_lists[treatment].append(x)
                        
                        y = np.array([row[stat]])
                        train_y_lists[treatment].append(y)
             
            for treatment in fitting.treatments:
                train_x_actual[treatment] = np.vstack(train_x_lists[treatment])
                train_y_actual[treatment] = np.vstack([arr.reshape((1, 1)) for arr in train_y_lists[treatment]])
            
            # TEST
            
            test_x_lists = {}
            test_y_lists = {}
            
            test_x_actual = {}
            test_y_actual = {}
            
            for treatment in fitting.treatments:
                test_x_lists[treatment] = []
                test_y_lists[treatment] = []
                  
            print('TEST')
            print('num of rows: {}'.format(len(df_test)))
            
            # build labels (for error checking later)
            test_y_labels = []
            
            # some problem with indexes - not sure what yet
            count = 0
            for index, row in df_test.iterrows():
                if count % 500 == 0:
                    print(count)
                    print(time.time())
                count += 1
                
                test_y_labels.append('{}_{}'.format(row['defense'], row['player']))
                
                # base arr is same for all treatments
                triangle_list = fitting.generate_triangles(G_full, row['defense'], row['player'], stat)
                triangle_arr = fitting.sort_triangles(triangle_list)
                
                for treatment in fitting.treatments:
                    x = fitting.route_treatment(triangle_arr, treatment)
                
                    # only add x,y to list if x actually returned value
                    if type(x) == np.ndarray:
                        test_x_lists[treatment].append(x)
                        
                        y = np.array([row[stat]])
                        test_y_lists[treatment].append(y)
             
            for treatment in fitting.treatments:
                test_x_actual[treatment] = np.vstack(test_x_lists[treatment])
                test_y_actual[treatment] = np.vstack([arr.reshape((1, 1)) for arr in test_y_lists[treatment]])
              
            
            # pred dict to store
            lin_reg_y_pred = {}
            random_forest_y_pred = {}
            
                    
            # train models and use to predict test_y
            for treatment in fitting.treatments:
                print('fitting linreg {}'.format(treatment))
                print(time.time())
            
                lin_reg.fit(train_x_actual[treatment], train_y_actual[treatment])
            
                print(time.time())
            
                lin_reg_y_pred[treatment] = lin_reg.predict(test_x_actual[treatment])
                
                print('fitting rf')
                print(time.time())
            
                random_forest.fit(train_x_actual[treatment], train_y_actual[treatment].ravel())
            
                print(time.time())
            
                random_forest_y_pred[treatment] = random_forest.predict(test_x_actual[treatment])
                
            
            # SAVE RESULTS
            # build dict to populate df
            data_dict = {}
            data_dict['label'] = test_y_labels
            
            for treatment in fitting.treatments:
                data_dict['linreg_{}'.format(treatment)] = lin_reg_y_pred[treatment].ravel()
                data_dict['rf_{}'.format(treatment)] = random_forest_y_pred[treatment]
                data_dict['actual_{}'.format(treatment)] = test_y_actual[treatment].ravel()

            
            # build df
            results_df = pd.DataFrame(data=data_dict)
            
            # save df
            file_path = 'results/{}{}_{}.csv'.format(stat, year, i)
            
            results_df.to_csv(file_path, index=False)
            
            print('results saved to {}'.format(file_path))

