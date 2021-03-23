# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 10:47:36 2021

@author: richie bao -workshop-LA-UP_IIT
"""
import network_structure_bus_subway as nsbs
import database as db
import networkx as nx
import os,sys
import matplotlib.pyplot as plt
import pandas as pd
import pickle
plt.rcParams['font.sans-serif'] = ['DengXian'] # 指定默认字体 'KaiTi','SimHei'
plt.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题

nanjing_epsg=32650 #Nanjing 

def population_flowOverNetwork(population,stop_geometries,all_shortest_length_dict_):
    from shapely.ops import nearest_points
    import pandas as pd
    from shapely.geometry import MultiPoint
    from tqdm import tqdm
    import pickle
    import copy
    
    population['geo_string']=population.geometry.apply(lambda row:"{}".format(row))
    population_dict=dict(zip(population.geo_string,population.Population))
    # print(population_dict)
    nearest_population_weight={}
    def select_rows(row):
        st_value_df_selection_row=st_value_df[st_value_df["geometry"]==nearest_points(row,MultiPoint(st_value_df.geometry.to_list()))[1]]
        # print(st_value_df_selection_row.time_cost.values[0])
        return st_value_df_selection_row.time_cost.values[0]        
    
    for st_key,st_value in tqdm(all_shortest_length_dict_.items()):
        # print(st_key,st_value)
        st_value_df=pd.DataFrame.from_dict(st_value,orient='index').reset_index().rename(columns={'index':'stop',0:'time_cost'})        
        st_value_df['geometry']=st_value_df.stop.apply(lambda row:stop_geometries[row])
        # print(st_value_df)
        
        #A-stops-->population deprecated
        # st_value_df['nearest_population']=st_value_df.geometry.apply(lambda row:population_dict["{}".format(nearest_points(row,MultiPoint(population.geometry.to_list()))[1])])
        # st_value_df['nearest_population_weight']=st_value_df.apply(lambda row:float(row.nearest_population)/(row.time_cost+0.1),axis=1)
        # nearest_population_weight[st_key]=st_value_df
        
        #B-population-->stops
        population_flow=copy.deepcopy(population)
        population_flow['stop_geometry']=population_flow.geometry.apply(lambda row:nearest_points(row,MultiPoint(st_value_df.geometry.to_list()))[1])
        population_flow['time_cost']=population_flow.geometry.apply(select_rows)
        population_flow['flow_weight']=population_flow.apply(lambda row:float(row.Population)/(row.time_cost+0.1),axis=1)
        nearest_population_weight[st_key]=population_flow
        # print(population_flow)
        # break
        
    with open('./results/ nearest_population_weight.pkl','wb') as f:
        pickle.dump(nearest_population_weight,f)
        
    return nearest_population_weight       


def dicts_merge(dict1,dict2):
    merged_dict={**dict1,**dict2}
    return merged_dict

def starting_stops_population_weights(nearest_population_weight,stop_geometries,epsg):
    from tqdm import tqdm
    import pandas as pd
    import geopandas as gpd
    import pyproj
    
    starting_stops_pop_weight={}
    i=0
    for starting_stops_Uid,n_pop_weight in tqdm(nearest_population_weight.items()):
        flow_weight_describe=n_pop_weight['flow_weight'].describe().to_dict()
        flow_weight_describe.update({"geometry":stop_geometries[starting_stops_Uid]})
        # print(flow_weight_describe)
        starting_stops_pop_weight[starting_stops_Uid]=flow_weight_describe        
        # if i==3:break
        # i+=1
        
    starting_stops_pop_weight_df=pd.DataFrame.from_dict(starting_stops_pop_weight, orient='index')  
    starting_stops_pop_weight_df.reset_index(inplace=True)
    starting_stops_pop_weight_df.rename(columns={'index':'stop_Uid'},inplace=True)
    crs=pyproj.CRS('EPSG:{}'.format(nanjing_epsg))
    starting_stops_pop_weight_gdf=gpd.GeoDataFrame(starting_stops_pop_weight_df,geometry='geometry',crs=crs)
    
    return starting_stops_pop_weight_gdf


if __name__=="__main__":  
    G_subway_bus_transfer=nx.read_gpickle("./results/G_subway_bus_transfer.gpickle")
    start_stops_PointUid=nsbs.postSQL2gpd(table_name='start_stops_fre_gdf',geom_col='geometry',myusername='postgres',mypassword='123456',mydatabase='workshop-LA-UP_IIT')
    start_stops_PointUid.plot()
    
    all_shortest_path_dict_,all_shortest_length_dict_=nsbs.bus_shortest_paths(G_subway_bus_transfer,start_stops_PointUid.PointUid)
    
    shortest_routes_fre_df_,shortest_df_dict_=nsbs.bus_service_index(G_subway_bus_transfer,all_shortest_path_dict_,all_shortest_length_dict_) 
    paths_0_=list(shortest_df_dict_.values())[0]
    edges_dict_=nsbs.G_draw_paths_composite(G_subway_bus_transfer,paths_0_)    

    population=nsbs.postSQL2gpd(table_name='population',geom_col='geometry',myusername='postgres',mypassword='123456',mydatabase='workshop-LA-UP_IIT')
    population.plot()
    
    bus_stations=nsbs.postSQL2gpd(table_name='bus_stations',geom_col='geometry',myusername='postgres',mypassword='123456',mydatabase='workshop-LA-UP_IIT')
    subway_stations=nsbs.postSQL2gpd(table_name='subway_stations',geom_col='geometry',myusername='postgres',mypassword='123456',mydatabase='workshop-LA-UP_IIT')
    bus_stations_dict=dict(zip(bus_stations.PointUid,bus_stations.geometry))
    subway_stations_dict=dict(zip(subway_stations.PointUid,subway_stations.geometry))
    stop_geometries=dicts_merge(bus_stations_dict,subway_stations_dict)
    
    nearest_population_weight=population_flowOverNetwork(population,stop_geometries,all_shortest_length_dict_)

    with open('./results/ nearest_population_weight.pkl','rb') as f:
        nearest_population_weight=pickle.load(f)
        
    starting_stops_population_weights_gdf=starting_stops_population_weights(nearest_population_weight,stop_geometries,nanjing_epsg)
    starting_stops_population_weights_gdf.plot(column='mean',cmap='afmhot',markersize=8)
    db.gpd2postSQL(starting_stops_population_weights_gdf,table_name='starting_stops_pop_weights',myusername='postgres',mypassword='123456',mydatabase='workshop-LA-UP_IIT')
