# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 10:31:37 2021

@author: richie bao -workshop-LA-UP_IIT
"""
import networkx as nx
import numpy as np

def postSQL2gpd(table_name,geom_col='geometry',**kwargs):
    from sqlalchemy import create_engine
    import geopandas as gpd
    
    engine=create_engine("postgres://{myusername}:{mypassword}@localhost:5432/{mydatabase}".format(myusername=kwargs['myusername'],mypassword=kwargs['mypassword'],mydatabase=kwargs['mydatabase']))  
    gdf=gpd.read_postgis(table_name, con=engine,geom_col=geom_col)
    print("_"*50)
    print('The data has been read from PostSQL database...')    
    return gdf
 
flatten_lst=lambda lst: [m for n_lst in lst for m in flatten_lst(n_lst)] if type(lst) is list else [lst]    
 
def bus_network(b_centroid_,bus_stations_,bus_routes_,**kwargs): #
    import copy
    import pandas as pd
    import networkx as nx
    from shapely.ops import nearest_points
    from shapely.ops import substring
    from tqdm import tqdm
    
    #compute the distance between the site centroid and each bus station and get the nearest ones by given threshold
    bus_stations=copy.deepcopy(bus_stations_)
    bus_stations['center_distance']=bus_stations.geometry.apply(lambda row:row.distance(b_centroid_.geometry.values[0]))
    # bus_stations.sort_values(by=['center_distance'],inplace=True)
    # print(bus_stations)
    start_stops=bus_stations[bus_stations.center_distance<=kwargs['start_stops_distance']]
    start_stops_lineUID=start_stops.LineUid.unique()   
    start_stops_PointUid=start_stops.PointUid.unique()   
    
    #build bus stations network
    bus_staions_routes=pd.merge(bus_stations,bus_routes,on='LineUid')
    bus_staions_routes_idx_LineUid=bus_staions_routes.set_index('LineUid',append=True,drop=False)    
    
    lines_group_list=[]
    s_e_nodes=[]
    # i=0
    for LineUid,sub_df in tqdm(bus_staions_routes_idx_LineUid.groupby(level=1)):
        # print(sub_df)
        # print(sub_df.columns)
        sub_df['nearestPts']=sub_df.apply(lambda row:nearest_points(row.geometry_y,row.geometry_x)[0],axis=1)
        sub_df['project_norm']=sub_df.apply(lambda row:row.geometry_y.project(row.nearestPts,normalized=True),axis=1)
        sub_df.sort_values(by='project_norm',inplace=True)
        sub_df['order_idx']=range(1,len(sub_df)+1)
        # station_geometries=sub_df.geometry_x.to_list()
        project=sub_df.project_norm.to_list()
        sub_df['second_project']=project[1:]+project[:1]
        
        PointName=sub_df.PointName.to_list()
        sub_df['second_PointName']=PointName[1:]+PointName[:1]
        PointUid=sub_df.PointUid.to_list()
        sub_df['second_PointUid']= PointUid[1:]+ PointUid[:1]
        
        sub_df['substring']=sub_df.apply(lambda row:substring(row.geometry_y,row.project_norm,row.second_project,normalized=True),axis=1)
        sub_df['forward_length']=sub_df.apply(lambda row:row.substring.length,axis=1)
        
        sub_df['edges']=sub_df.apply(lambda row:[(row.PointUid,row.second_PointUid),(row.second_PointUid,row.PointUid)],axis=1)
        
        lines_group_list.append(sub_df)
        s_e_nodes.append(sub_df.edges.to_list()[-1][0])
        
        # print(i)
        # i+=1
    lines_df4G=pd.concat(lines_group_list)
    
    # G=nx.Graph()
    G=nx.from_pandas_edgelist(df=lines_df4G,source='PointUid',target='second_PointUid',edge_attr=['PointName','second_PointName','forward_length','geometry_x'])
    for idx,row in lines_df4G.iterrows():
        G.nodes[row['PointUid']]['position']=(row.geometry_x.x,row.geometry_x.y)
    
    return G,s_e_nodes,start_stops_PointUid,lines_df4G

def transfer_stations_network(station_geometries_df,transfer_distance,transfer_weight_ratio): 
    import copy
    from tqdm import tqdm
    import pandas as pd
    import networkx as nx
    
    transfer_df_list=[]
    station_geometries_dict=station_geometries_df.to_dict('record')
    # i=0
    for pt in tqdm(station_geometries_dict):
        station_geometries_df_=copy.deepcopy(station_geometries_df)
        station_geometries_df_['distance']=station_geometries_df_.geometry_x.apply(lambda row:row.distance(pt['geometry_x']))
        
        transfer_df=station_geometries_df_[station_geometries_df_.distance<=transfer_distance]
        transfer_df=transfer_df[transfer_df.distance!=0]
        transfer_df.drop_duplicates(subset='PointUid',keep='first',inplace=True)        
        
        transfer_df['source_station']=pt['PointUid']
        transfer_df['forward_length']=transfer_df.distance*transfer_weight_ratio
        transfer_df=transfer_df[transfer_df.LineUid!=pt['LineUid']]        
  
        transfer_df_list.append(transfer_df)
        
        # if i==100:break
        # i+=1
       
    transfer_df_concat=pd.concat(transfer_df_list)
    # print(transfer_df_concat)
    G=nx.from_pandas_edgelist(df=transfer_df_concat,source='source_station',target='PointUid',edge_attr=['forward_length'])

    return  G

def G_draw(G,layout='spring_layout',node_color=None,node_size=None,figsize=(30, 30),font_size=12,edge_color=None):    
    import matplotlib
    import matplotlib.pyplot as plt
    import networkx as nx
    '''
    function - To show a networkx graph
    '''
    #解决中文显示问题
    plt.rcParams['font.sans-serif'] = ['DengXian'] # 指定默认字体 'KaiTi','SimHei'
    plt.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题
    fig, ax = plt.subplots(figsize=figsize)
    #nx.draw_shell(G, with_labels=True)
    layout_dic={
        'spring_layout':nx.spring_layout,   
        'random_layout':nx.random_layout,
        'circular_layout':nx.circular_layout,
        'kamada_kawai_layout':nx.kamada_kawai_layout,
        'shell_layout':nx.shell_layout,
        'spiral_layout':nx.spiral_layout,
    }

    nx.draw(G,nx.get_node_attributes(G,'position'),with_labels=False,node_color=node_color,node_size=node_size,font_size=font_size,edge_color=edge_color)  #nx.draw(G, pos, font_size=16, with_labels=False)


def bus_shortest_paths(G,start_stops_PointUid):
    from tqdm import tqdm
    
    all_shortest_length_dict={}
    all_shortest_path_dict={}
    start_stops_PointUi_list=start_stops_PointUid.tolist()
    for stop in tqdm(start_stops_PointUi_list):
        shortest_path=nx.shortest_path(G, source=stop,weight="forward_length")
        shortest_length=nx.shortest_path_length(G, source=stop,weight="forward_length")
        all_shortest_length_dict[stop]=shortest_length
        all_shortest_path_dict[stop]=shortest_path
        # break

    return all_shortest_path_dict,all_shortest_length_dict




if __name__=="__main__":  
    b_centroid=postSQL2gpd(table_name='b_centroid',geom_col='geometry',myusername='postgres',mypassword='123456',mydatabase='workshop-LA-UP_IIT')

    # population=postSQL2gpd(table_name='population',geom_col='geometry',myusername='postgres',mypassword='123456',mydatabase='workshop-LA-UP_IIT')
    bus_stations=postSQL2gpd(table_name='bus_stations',geom_col='geometry',myusername='postgres',mypassword='123456',mydatabase='workshop-LA-UP_IIT')
    bus_routes=postSQL2gpd(table_name='bus_routes',geom_col='geometry',myusername='postgres',mypassword='123456',mydatabase='workshop-LA-UP_IIT')
    G_bus_stations,s_e_nodes,start_stops_PointUid,lines_df4G=bus_network(b_centroid,bus_stations,bus_routes,start_stops_distance=500) #bus_network_structure
    G_bus_stations.remove_edges_from(s_e_nodes)
    G_draw(G_bus_stations,edge_color=list(nx.get_edge_attributes(G_bus_stations, 'forward_length').values()))
    
    station_geometries_df=lines_df4G[['PointUid','geometry_x','LineUid']] 
    # forward_length_mean=np.mean(np.array(list(nx.get_edge_attributes(G_bus, 'forward_length').values())))  
    G_bus_transfer=transfer_stations_network(station_geometries_df,transfer_distance=200,transfer_weight_ratio=2)    

    G_bus=nx.compose(G_bus_stations,G_bus_transfer)
    G_draw(G_bus,edge_color=list(nx.get_edge_attributes(G_bus, 'forward_length').values()))
    
    all_shortest_path_dict,all_shortest_length_dict=bus_shortest_paths(G_bus,start_stops_PointUid)
    
    a=list(all_shortest_path_dict.values())[10]
    b=list(a.values())[3000]
    H=G_bus.subgraph(b)
    G_draw(H,edge_color=list(nx.get_edge_attributes(H, 'forward_length').values()))
    
    