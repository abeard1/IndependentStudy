import operator
import numpy as np
import networkx as nx


def select_top_m_gp(m, triangle_list):

    if len(triangle_list) < m:
        return None
        
    triangle_list.sort(key=operator.itemgetter(1), reverse=True)
        
    top_m = [x[0] for x in triangle_list[:m]]

    return np.array(top_m)


def select_top_5_gp(triangle_list):
    return select_top_m_gp(5, triangle_list)


def select_top_10_gp(triangle_list):
    return select_top_m_gp(10, triangle_list)


def select_m_percentiles(m, triangle_list):

    if len(triangle_list) < m:
        return None
    
    just_preds = [x[0] for x in triangle_list]

    just_preds.sort()

    jump = int(100/m)
    percentiles = list(range(int(jump/2), 100, jump))

    vals = [np.percentile(just_preds, p) for p in percentiles]

    return np.array(vals)


def select_5_percentiles(triangle_list):
    return select_m_percentiles(5, triangle_list)


def select_10_percentiles(triangle_list):
    return select_m_percentiles(10, triangle_list)


def select_avg(triangle_list):
    if len(triangle_list) < 1:
        return None

    just_preds = [x[0] for x in triangle_list]

    val = sum(just_preds) / len(just_preds)
    arr = np.array([val])
    return arr.reshape((1,1))

def num_triangles(G, defense, player):
    num_triangles = 0

    for player_other in G[defense]:
        if player_other != player:
            for defense_other in G[player_other]:
                if defense_other != defense and player in G[defense_other]:
                        num_triangles += 1

    return num_triangles

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

selection_methods = [select_top_5_gp, select_top_10_gp, select_5_percentiles, select_10_percentiles, select_avg]


