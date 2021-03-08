# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 10:31:37 2021

@author: richie bao -workshop-LA-UP_IIT

speed: subway-about 60 k/h;bus-about 45-50 (45) k/h;walk-about 3.6-7.2 (5)k/m. 1k(ilometer)=1000m
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
 
def bus_network(b_centroid_,bus_stations_,bus_routes_,speed,**kwargs): #
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
    bus_staions_routes=pd.merge(bus_stations,bus_routes_,on='LineUid')
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
        sub_df['time_cost']=sub_df.apply(lambda row:row.forward_length/(speed*1000)*60,axis=1)
        
        sub_df['edges']=sub_df.apply(lambda row:[(row.PointUid,row.second_PointUid),(row.second_PointUid,row.PointUid)],axis=1)
        
        lines_group_list.append(sub_df)
        s_e_nodes.append(sub_df.edges.to_list()[-1][0])
        
        # print(i)
        # i+=1
    lines_df4G=pd.concat(lines_group_list)
    
    # G=nx.Graph()
    G=nx.from_pandas_edgelist(df=lines_df4G,source='PointUid',target='second_PointUid',edge_attr=['PointName','second_PointName','forward_length','geometry_x','time_cost'])
    for idx,row in lines_df4G.iterrows():
        G.nodes[row['PointUid']]['position']=(row.geometry_x.x,row.geometry_x.y)
        G.nodes[row['PointUid']]['station_name']=row.PointName        
    
    return G,s_e_nodes,start_stops_PointUid,lines_df4G

def transfer_stations_network(station_geometries_df,transfer_distance,transfer_weight_ratio,speed): 
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
        # print(transfer_df['forward_length'])
        transfer_df=transfer_df[transfer_df.LineUid!=pt['LineUid']]      
        # print(transfer_df)    
  
        transfer_df_list.append(transfer_df)
        
        # if i==1:break
        # i+=1
       
    transfer_df_concat=pd.concat(transfer_df_list)
    transfer_df_concat['time_cost']=transfer_df_concat.apply(lambda row:row.forward_length/(speed*1000)*60,axis=1)
    # print(transfer_df_concat)
    G=nx.from_pandas_edgelist(df=transfer_df_concat,source='source_station',target='PointUid',edge_attr=['forward_length','time_cost'])

    return  G,transfer_df_concat

def G_draw(G,layout='spring_layout',node_color=None,node_size=None,figsize=(30, 30),font_size=12,edge_color=None,labels=None,with_labels=False):    
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

    nx.draw(G,nx.get_node_attributes(G,'position'),with_labels=with_labels,labels=labels,node_color=node_color,node_size=node_size,font_size=font_size,edge_color=edge_color)  #nx.draw(G, pos, font_size=16, with_labels=False)


def bus_shortest_paths(G,start_stops_PointUid):
    from tqdm import tqdm
    
    all_shortest_length_dict={}
    all_shortest_path_dict={}
    start_stops_PointUi_list=start_stops_PointUid.tolist()
    for stop in tqdm(start_stops_PointUi_list):
        shortest_path=nx.shortest_path(G, source=stop,weight="time_cost")
        shortest_length=nx.shortest_path_length(G, source=stop,weight="time_cost")
        all_shortest_length_dict[stop]=shortest_length
        all_shortest_path_dict[stop]=shortest_path
        # break

    return all_shortest_path_dict,all_shortest_length_dict


def bus_service_index(G,all_shortest_path_dict,all_shortest_length_dict):
    import pandas as pd
    from tqdm import tqdm
    
    start,end,step=0,25,5      #start,end,step=0,10000,2000
    def partition(value,start,end,step):
        import numpy as np
        ranges_list=list(range(start,end+step,step))
        ranges=[(ranges_list[i],ranges_list[i+1]) for i in  range(len(ranges_list)-1)]

        for r in ranges:
            # print(r,value)
            if r[0]<=value<r[1] :    
                return r[1]
    
    shortest_df_dict={}
    shortest_routes_fre={}
    for start_stop,shortest_length in tqdm(all_shortest_length_dict.items()):
        # print(start_stop)
        shortest_df=pd.DataFrame.from_dict(shortest_length, orient='index',columns=['duration'])  
        shortest_df['path']=shortest_df.index.map(all_shortest_path_dict[start_stop])
        shortest_df['range']=shortest_df.duration.apply(partition,args=(start,end,step))
        # print(shortest_df.range)
        ranges_frequency=shortest_df['range'].value_counts()
        shortest_routes_fre[start_stop]=ranges_frequency.to_dict()
        # print(ranges_frequency.to_dict())
        shortest_df_dict[start_stop]=shortest_df

    shortest_routes_fre_df=pd.DataFrame.from_dict(shortest_routes_fre,orient='index')
    
    return shortest_routes_fre_df,shortest_df_dict


def G_draw_path(G,paths,figsize=(30, 30)):
    import networkx as nx
    import matplotlib.pyplot as plt
    import random
    fig, ax=plt.subplots(figsize=figsize)
    
    G_p=G.copy()
    G_p.remove_edges_from(list(G_p.edges()))
    G_p_range=nx.Graph(G_p.subgraph(flatten_lst(paths)))
    
    edges=[]
    for r in paths:
        path_edges=[(r[n],r[n+1]) for n in range(len(r)-1)]
        G_p_range.add_edges_from(path_edges)
        edges.append(path_edges)
    print("Graph has %d nodes with %d paths(edges)" %(G_p_range.number_of_nodes(), G_p_range.number_of_edges()))   
    pos=nx.get_node_attributes(G_p_range,'position')
    nx.draw_networkx_nodes(G_p_range,pos=pos,node_size=300) #,with_labels=True,labels=nx.get_node_attributes(G_p_range,'station_name'),font_size=20
    # nx.draw_networkx_labels(G_p_range,pos=pos,labels=nx.get_node_attributes(G_p_range,'station_name'),font_size=8)
    # colors = ['r', 'b', 'y']
    # linewidths = [20,10,5]    
    for ctr, edgelist in enumerate(edges):
        color=(random.random(), random.random(),random.random())
        nx.draw_networkx_edges(G_p_range,pos=pos,edgelist=edgelist,edge_color=color) #edge_color = colors[ctr], width=linewidths[ctr]
    # plt.savefig('G_p_range.png')

def G_draw_range_paths(G,range_paths,figsize=(30, 30)):
   
    ranges=range_paths.range.unique()
    for i in ranges:
        range_paths_range=range_paths[range_paths.range==i]
        path_edges=range_paths_range.path.to_list()
        G_draw_path(G,path_edges)

def G_draw_paths_composite(G,range_paths,figsize=(30, 30)):
    import networkx as nx
    import matplotlib.pyplot as plt
    import random
    from collections import defaultdict
    
    fig, ax=plt.subplots(figsize=figsize)
    
    ranges=range_paths.range.unique()
    print(ranges)
    path_edges_dict={}
    for i in ranges:
        range_paths_range=range_paths[range_paths.range==i]
        path_edges=range_paths_range.path.to_list()    
        path_edges_dict[i]=path_edges
    
    G_p=G.copy()
    G_p.remove_edges_from(list(G_p.edges()))
    G_p_range=nx.Graph(G_p.subgraph(flatten_lst(list(path_edges_dict.values()))))
    
    edges_dict=defaultdict(list)
    for k in path_edges_dict.keys():
        for r in path_edges_dict[k]:
            path_edges=[(r[n],r[n+1]) for n in range(len(r)-1)]
            G_p_range.add_edges_from(path_edges)
            edges_dict[k].append(path_edges)
    print("Graph has %d nodes with %d paths(edges)" %(G_p_range.number_of_nodes(), G_p_range.number_of_edges()))   
        
    pos=nx.get_node_attributes(G_p_range,'position')
    nx.draw_networkx_nodes(G_p_range,pos=pos,node_size=100) #,with_labels=True,labels=nx.get_node_attributes(G_p_range,'station_name'),font_size=20
    nx.draw_networkx_labels(G_p_range,pos=pos,labels=nx.get_node_attributes(G_p_range,'station_name'),font_size=8)
    linewidths=[5,10,15,20,25]
    linewidths.reverse()
    colors=['red','blue','green']
    for j,k in enumerate(reversed(list(edges_dict.keys()))):
        # color=(random.random(), random.random(),random.random())
        for i, edgelist in enumerate(edges_dict[k]):            
            nx.draw_networkx_edges(G_p_range,pos=pos,edgelist=edgelist,edge_color=colors[j],width=linewidths[j]) #edge_color = colors[ctr], width=linewidths[ctr]
    # plt.savefig('G_p_range.png')    
    
    return edges_dict
    
'''
def subway_network(b_centroid_,subway_stations_,subway_routes_,speed,**kwargs): #
    import copy
    import pandas as pd
    import networkx as nx
    from shapely.ops import nearest_points
    from shapely.ops import substring
    from tqdm import tqdm
    
    #compute the distance between the site centroid and each bus station and get the nearest ones by given threshold
    subway_stations=copy.deepcopy(subway_stations_)
    subway_stations['center_distance']=subway_stations.geometry.apply(lambda row:row.distance(b_centroid_.geometry.values[0]))
    # subway_stations.sort_values(by=['center_distance'],inplace=True)
    # print(subway_stations)
    start_stops=subway_stations[subway_stations.center_distance<=kwargs['start_stops_distance']]
    start_stops_lineUID=start_stops.LineUid.unique()   
    start_stops_PointUid=start_stops.PointUid.unique()   
    # print(start_stops_PointUid)
    #build bus stations network
    subway_staions_routes=pd.merge(subway_stations,subway_routes_,on='LineUid')
    subway_staions_routes_idx_LineUid=subway_staions_routes.set_index('LineUid',append=True,drop=False)    
    
    lines_group_list=[]
    s_e_nodes=[]
    # i=0
    for LineUid,sub_df in tqdm(subway_staions_routes_idx_LineUid.groupby(level=1)):
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
        sub_df['time_cost']=sub_df.apply(lambda row:row.forward_length/(speed*1000)*60,axis=1)
        
        sub_df['edges']=sub_df.apply(lambda row:[(row.PointUid,row.second_PointUid),(row.second_PointUid,row.PointUid)],axis=1)
        
        lines_group_list.append(sub_df)
        s_e_nodes.append(sub_df.edges.to_list()[-1][0])
        
        # print(i)
        # i+=1
    lines_df4G=pd.concat(lines_group_list)
    
    # G=nx.Graph()
    G=nx.from_pandas_edgelist(df=lines_df4G,source='PointUid',target='second_PointUid',edge_attr=['PointName','second_PointName','forward_length','geometry_x','time_cost'])
    for idx,row in lines_df4G.iterrows():
        G.nodes[row['PointUid']]['position']=(row.geometry_x.x,row.geometry_x.y)
        G.nodes[row['PointUid']]['station_name']=row.PointName        
    
    return G,s_e_nodes,start_stops_PointUid,lines_df4G
'''

def transfer_stations_network_subway_bus(bus_df,subway_df,transfer_distance,transfer_weight_ratio,speed): 
    import copy
    from tqdm import tqdm
    import pandas as pd
    import networkx as nx
    
    transfer_df_list=[]
    bus_stations_=bus_df[['PointUid','geometry_x','LineUid']]
    subway_station_=subway_df[['PointUid','geometry_x','LineUid']]
    subway_station=subway_station_.to_dict('recorde')
    for pt in tqdm(subway_station):
        # print(pt)
        bus_stations=copy.deepcopy(bus_stations_)
        bus_stations['distance']=bus_stations.geometry_x.apply(lambda row:row.distance(pt['geometry_x']))
        # print(bus_stations)
        transfer_df=bus_stations[bus_stations.distance<=transfer_distance]
        transfer_df=transfer_df[transfer_df.distance!=0]
        transfer_df.drop_duplicates(subset='PointUid',keep='first',inplace=True)   
        # print(transfer_df)
        
        transfer_df['source_station']=pt['PointUid']
        transfer_df['forward_length']=transfer_df.distance*transfer_weight_ratio
        transfer_df_list.append(transfer_df)
   
        # break
    transfer_df_concat=pd.concat(transfer_df_list)
    transfer_df_concat['time_cost']=transfer_df_concat.apply(lambda row:row.forward_length/(speed*1000)*60,axis=1)
    G=nx.from_pandas_edgelist(df=transfer_df_concat,source='source_station',target='PointUid',edge_attr=['forward_length','time_cost'])
    
    
    return G,transfer_df_concat


if __name__=="__main__":  
    #bus_network
    b_centroid=postSQL2gpd(table_name='b_centroid',geom_col='geometry',myusername='postgres',mypassword='123456',mydatabase='workshop-LA-UP_IIT')

    # population=postSQL2gpd(table_name='population',geom_col='geometry',myusername='postgres',mypassword='123456',mydatabase='workshop-LA-UP_IIT')
    bus_stations=postSQL2gpd(table_name='bus_stations',geom_col='geometry',myusername='postgres',mypassword='123456',mydatabase='workshop-LA-UP_IIT')
    bus_routes=postSQL2gpd(table_name='bus_routes',geom_col='geometry',myusername='postgres',mypassword='123456',mydatabase='workshop-LA-UP_IIT')
    G_bus_stations,s_e_nodes,start_stops_PointUid,lines_df4G=bus_network(b_centroid,bus_stations,bus_routes,speed=45,start_stops_distance=500) #bus_network_structure
    G_bus_stations.remove_edges_from(s_e_nodes)
    G_draw(G_bus_stations,edge_color=list(nx.get_edge_attributes(G_bus_stations, 'time_cost').values()),labels=nx.get_node_attributes(G_bus_stations,'station_name'),font_size=20)
    
    station_geometries_df=lines_df4G[['PointUid','geometry_x','LineUid']] 
    # forward_length_mean=np.mean(np.array(list(nx.get_edge_attributes(G_bus, 'forward_length').values())))  
    G_bus_transfer,transfer_df_concat=transfer_stations_network(station_geometries_df,transfer_distance=200,transfer_weight_ratio=9,speed=45)   #2 

    G_bus=nx.compose(G_bus_stations,G_bus_transfer)
    edge_color_=list(nx.get_edge_attributes(G_bus, 'time_cost').values()) #'forward_length'
    # edge_color=[i/max(edge_color_) for i in edge_color_]
    G_draw(G_bus,edge_color=edge_color_,labels=nx.get_node_attributes(G_bus_stations,'station_name'),font_size=20)
    
    all_shortest_path_dict,all_shortest_length_dict=bus_shortest_paths(G_bus,start_stops_PointUid)
    
    a=list(all_shortest_path_dict.values())[10]
    b=list(a.values())[3000]
    H=G_bus.subgraph(b)
    G_draw(H,edge_color=list(nx.get_edge_attributes(H, 'time_cost').values()),labels=nx.get_node_attributes(H,'station_name'),font_size=20)
    
    
    shortest_routes_fre_df,shortest_df_dict=bus_service_index(G_bus,all_shortest_path_dict,all_shortest_length_dict)   
    shortest_routes_fre_df_statistics=shortest_routes_fre_df.describe(include='all')
    
    paths_0=list(shortest_df_dict.values())[0]
    G_draw_range_paths(G_bus,paths_0)
    edges_dict=G_draw_paths_composite(G_bus,paths_0)
    # paths_0_1000=paths_0[paths_0.length<=10000]
    # path_0_edges=paths_0_1000.path.to_list()
    
    nx.write_gpickle(G_bus, "./results/G_bus.gpickle")

    #subway_network
    subway_stations=postSQL2gpd(table_name='subway_stations',geom_col='geometry',myusername='postgres',mypassword='123456',mydatabase='workshop-LA-UP_IIT')
    subway_routes=postSQL2gpd(table_name='subway_lines',geom_col='geometry',myusername='postgres',mypassword='123456',mydatabase='workshop-LA-UP_IIT')

    G_subway_stations,subway_s_e_nodes,subway_start_stops_PointUid,subway_lines_df4G=bus_network(b_centroid,subway_stations,subway_routes,speed=60*2,start_stops_distance=1200) #bus_network_structure
    G_subway_stations.remove_edges_from(subway_s_e_nodes)
    G_draw(G_subway_stations,edge_color=list(nx.get_edge_attributes(G_subway_stations, 'time_cost').values()),labels=nx.get_node_attributes(G_subway_stations,'station_name'),font_size=20)

    #transfer_subway-bus
    G_subway_bus_transfer,subway_bus_transfer_df_concat=transfer_stations_network_subway_bus(bus_df=lines_df4G,subway_df=subway_lines_df4G,transfer_distance=300,transfer_weight_ratio=9,speed=45)   #2 
    G_subway_bus=nx.compose(G_bus,G_subway_stations)
    G_subway_bus_transfer=nx.compose(G_subway_bus,G_subway_bus_transfer)

    all_shortest_path_dict_,all_shortest_length_dict_=bus_shortest_paths(G_subway_bus_transfer,start_stops_PointUid)
    shortest_routes_fre_df_,shortest_df_dict_=bus_service_index(G_subway_bus_transfer,all_shortest_path_dict_,all_shortest_length_dict_) 

    paths_0_=list(shortest_df_dict_.values())[0]
    edges_dict_=G_draw_paths_composite(G_subway_bus_transfer,paths_0_)

    nx.write_gpickle(G_subway_bus_transfer, "./results/G_subway_bus_transfer.gpickle")

    #Comparison of accessibility of bus stops adjacent to the site.
    import pandas as pd
    start_stops_PointUid_list=list(shortest_routes_fre_df_.index)
    start_stops_gdf=bus_stations[bus_stations.PointUid.isin(start_stops_PointUid_list)]
    shortest_routes_fre_df_resetIdx=shortest_routes_fre_df_.reset_index()
    shortest_routes_fre_df_resetIdx.rename(columns={'index':'PointUid'},inplace=True)
    start_stops_fre_gdf=pd.merge(start_stops_gdf,shortest_routes_fre_df_resetIdx,on='PointUid')
    
    import database
    database.gpd2postSQL(start_stops_fre_gdf,table_name='start_stops_fre_gdf',myusername='postgres',mypassword='123456',mydatabase='workshop-LA-UP_IIT')

