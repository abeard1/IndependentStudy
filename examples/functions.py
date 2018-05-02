import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def bipartite_draw(G, nodes=None, label_pos=0.4):
    temp_G = G.copy()
    
    if nodes:
        for n in G:
            if n not in nodes:
                temp_G.remove_node(n)
    
    # need this to draw weights later
    pos = nx.spring_layout(temp_G)

    # Separate by group (uses original graph because temp_G may not be able to differentiate)
    l, r = nx.bipartite.sets(G)
    pos = {}

    # Update position for node from each group
    pos.update((node, (1, index)) for index, node in enumerate(l))
    pos.update((node, (2, index)) for index, node in enumerate(r))

    nx.draw(temp_G, pos=pos, with_labels=True)
    nx.draw_networkx_edge_labels(temp_G, pos, label_pos=label_pos)
        
    plt.show()


def generate_triangle_dict(G, defense, player, attr):
    
    edge_list = list(G.edges())
    triangle_dict = {}
    
    for player_other in G[defense]:
        if player_other != player:
            for defense_other in G[player_other]:
                if defense_other != defense and player in G[defense_other]:
                    
                    edge_key = (player_other, defense_other)
                    if edge_key not in edge_list:
                        edge_key = (defense_other, player_other)
                    
                    other_other = G[defense_other][player_other][attr]
                    this_other = G[defense_other][player][attr]
                    other_this = G[defense][player_other][attr]
                    
                    # cannot be 0 (div by 0)
                    if other_other == 0:
                        other_other = 1
		
                    predicted = (this_other/other_other) * other_this
                                        
                    triangle_dict[edge_key] = predicted
                        
    return triangle_dict  

def split_edge(edge):
    if 'Defense' in edge[0]:
        return (edge[0], edge[1])
    return (edge[1], edge[0])

def build_df_pred(G_full, G_train, df_orig, M, attr):
    df_pred = df_orig.copy()
    
    df_pred['actual'] = df_pred.apply(lambda row: actual_val(G_full, row, attr), axis=1)

    m_rows = []
    for index, row in df_pred.iterrows():
        tri_dict = generate_triangle_dict(G_train, row['defense'], row['player'], attr)
        m_row = m_arr(tri_dict, M)
        m_rows.append(m_row)

    m_mat = np.vstack(m_rows)

    columns = ['m{}'.format(m) for m in range(M)]
        
    m_df = pd.DataFrame(data=m_mat, columns=columns)
        
    df_pred = pd.concat([df_pred, m_df], axis=1)

    return df_pred
    
def actual_val(G_full, row, attr):
    defense = row['defense']
    player = row['player']

    return G_full[defense][player][attr] 

# note: this will be modified for other selection/ranking strategies
# right now: smallest and largest
def m_arr(tri_dict, M):
    pred_list = list(tri_dict.values() )
    pred_list.sort()
    pred_list = [pred_list[0], pred_list[-1]]


    return np.array(pred_list)

