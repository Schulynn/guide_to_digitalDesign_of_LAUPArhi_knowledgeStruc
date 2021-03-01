# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 10:31:37 2021

@author: richie bao -workshop-LA-UP_IIT
"""


def postSQL2gpd(table_name,geom_col='geometry',**kwargs):
    from sqlalchemy import create_engine
    import geopandas as gpd
    engine=create_engine("postgres://{myusername}:{mypassword}@localhost:5432/{mydatabase}".format(myusername=kwargs['myusername'],mypassword=kwargs['mypassword'],mydatabase=kwargs['mydatabase']))  
    gdf=gpd.read_postgis(table_name, con=engine,geom_col=geom_col)
    print("_"*50)
    print('The data has been read from PostSQL database...')    
    return gdf


















if __name__=="__main__":    
    population=postSQL2gpd(table_name='population',geom_col='geometry',myusername='postgres',mypassword='123456',mydatabase='workshop-LA-UP_IIT')
    bus_stations=postSQL2gpd(table_name='bus_stations',geom_col='geometry',myusername='postgres',mypassword='123456',mydatabase='workshop-LA-UP_IIT')
    bus_routes=postSQL2gpd(table_name='bus_routes',geom_col='geometry',myusername='postgres',mypassword='123456',mydatabase='workshop-LA-UP_IIT')
