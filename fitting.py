import operator
import numpy as np
import networkx as nx

treatments = ['null', 'median_5', 'median_10', 'mean_5', 'mean_10']

def generate_triangles(G, defense, player, stat):
    
    triangle_list = []
    
    for player_other in G[defense]:
        if player_other != player:
            for defense_other in G[player_other]:
                if defense_other != defense and player in G[defense_other]:
                    
                    #edge_key = (player_other, defense_other)
                    #if edge_key not in edge_list:
                    #    edge_key = (defense_other, player_other)
                    
                    other_other = G[defense_other][player_other][stat]
                    this_other = G[defense_other][player][stat]
                    other_this = G[defense][player_other][stat]
                    
                    # 0 if other_other is 0
                    if other_other == 0:
                        predicted = 0
                    
                    else:
                        predicted = (this_other/other_other) * other_this
                    
                    avg_times_played = G[defense_other][player_other]['times_played'] * G[defense_other][player]['times_played'] * G[defense][player_other]['times_played']
                                        
                    #triangle_dict[edge_key] = predicted
                    
                    # triangle list is now list of tuples
                    triangle = (predicted, avg_times_played)
                    triangle_list.append(triangle)
                        
    return triangle_list 

def sort_triangles(triangle_list):
    triangle_list.sort(key=operator.itemgetter(1), reverse=True)

    # get rid of gp - no longer needed
    raw_list = [x[0] for x in triangle_list]

    return np.array(raw_list)

def split_chunks(triangle_arr, num_chunks):
    split = np.array_split(triangle_arr, num_chunks)

    return split

def chunk_medians(chunks):
    medians_list = [np.median(chunk) for chunk in chunks]
    return np.array(medians_list)

def chunk_means(chunks):
    means_list = [np.mean(chunk) for chunk in chunks]
    return np.array(means_list)

def route_treatment(triangle_arr, treatment):
    mean = np.mean(triangle_arr)
    median = np.median(triangle_arr)

    null_arr = np.array([mean, median])

    #if np.isnan(mean) or np.isnan(median):
    #    print('NAN FOUND')
    #    print(triangle_arr)
    #    return 'NAN'

    if treatment == 'null':
        return null_arr 

    if treatment == 'median_5':
        chunks = split_chunks(triangle_arr, 5)
        add = chunk_medians(chunks)
        return np.append(null_arr, add)
    elif treatment == 'median_10':  
        chunks = split_chunks(triangle_arr, 10)
        add = chunk_medians(chunks)
        return np.append(null_arr, add)  
    elif treatment == 'mean_5':  
        chunks = split_chunks(triangle_arr, 5)
        add = chunk_medians(chunks)
        return np.append(null_arr, add)  
    elif treatment == 'mean_10':
        chunks = split_chunks(triangle_arr, 5)
        add = chunk_medians(chunks)
        return np.append(null_arr, add)

    else:
        print('ERROR: invalid treatment string')

    
