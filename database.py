# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 13:12:28 2021

@author: richie bao -workshop-LA-UP_IIT
"""
import geopandas as gpd
import fiona,io
from tqdm import tqdm
import pyproj

# pd.set_option('display.max_columns', None)

nanjing_epsg=32650 #Nanjing
data_dic={
    'road_network':r'.\data\GIS\road Network Data OF Nanjing On 20190716.kml',
    'qingliangMountain_boundary':r'./data/GIS/QingliangMountain_boundary.kml',
    'building_footprint':r'./data/GIS/Nanjing Building footprint Data/NanjingBuildingfootprintData.shp',
    'bus_routes':r'./data/GIS/SHP data of Nanjing bus route and stations in December 2020/busRouteStations_20201218135814.shp',
    'bus_stations':r'./data/GIS/SHP data of Nanjing bus route and stations in December 2020/busRouteStations_20201218135812.shp',
    'subway_lines':r'./data/GIS/SHP of Nanjing subway station and line on 2020/SHP of Nanjing subway station and line on 2020 (2).shp',
    'subway_stations':r'./data/GIS/SHP of Nanjing subway station and line on 2020/SHP of Nanjing subway station and line on 2020.shp',
    'population':r'./data/GIS/SHP of population distribution in Nanjing in 2020/SHP of population distribution in Nanjing in 2020.shp',
    'taxi':r'./data/GIS/Nanjing taxi data in 2016',
    'POI':r"./data/GIS/Nanjing POI 201912.csv",
    'microblog':r'./data/GIS/Najing Metro Weibo publish.db',
    'bike_sharing':r'./data/GIS/One hundred thousand shared bikes.xls',
    'sentinel_2':r'C:\Users\richi\omen_richiebao\omen_IIIT\workshop_LA_UP_iit\data\RS\S2B_MSIL2A_20200819T024549_N0214_R132_T50SPA_20200819T045147.SAFE',
    }

class SQLite_handle():
    def __init__(self,db_file):
        self.db_file=db_file   
    
    def create_connection(self):
        import sqlite3
        from sqlite3 import Error
        """ create a database connection to a SQLite database """
        conn=None
        try:
            conn=sqlite3.connect(self.db_file)
            print('connected.',"SQLite version:%s"%sqlite3.version,)
        except Error as e:
            print(e)
        finally:
            if conn:
                conn.close()

def boundary_angularPts(bottom_left_lon,bottom_left_lat,top_right_lon,top_right_lat):
    bottom_left=(bottom_left_lon,bottom_left_lat)
    top_right=(top_right_lon,top_right_lat)
    boundary=[(bottom_left[0],bottom_left[1]),(top_right[0],bottom_left[1]),(top_right[0], top_right[1]),(bottom_left[0], top_right[1])]
    return boundary

def boundary_buffer_centroidCircle(kml_extent,proj_epsg,bounadry_type='buffer_circle',buffer_distance=1000):
    import pyproj
    from shapely.ops import transform
    from shapely.geometry import Point,LinearRing,Polygon
    import geopandas as gpd
    
    gpd.io.file.fiona.drvsupport.supported_drivers['KML'] = 'rw'
    boundary_gdf=gpd.read_file(kml_extent,driver='KML')
    # print(boundary_gdf)
    
    wgs84=pyproj.CRS('EPSG:4326')
    utm=pyproj.CRS('EPSG:{}'.format(proj_epsg))
    project=pyproj.Transformer.from_crs(wgs84, utm, always_xy=True).transform

    boundary_proj=transform(project,boundary_gdf.geometry.values[0])    
    
    if bounadry_type=='buffer_circle':
        b_centroid=boundary_proj.centroid
        b_centroid_buffer=b_centroid.buffer(buffer_distance)
        c_area=[b_centroid_buffer.area]
        gpd.GeoDataFrame({'area': c_area,'geometry':b_centroid_buffer},crs=utm).to_crs(wgs84).to_file('./data/GIS/b_centroid_buffer.shp')     
        
        b_centroid_gpd=gpd.GeoDataFrame({'x':[b_centroid.x],'y':[b_centroid.y],'geometry':[b_centroid]},crs=utm)# .to_crs(wgs84)
        gpd2postSQL(b_centroid_gpd,table_name='b_centroid',myusername='postgres',mypassword='123456',mydatabase='workshop-LA-UP_IIT')
        return b_centroid_buffer
    
    elif bounadry_type=='buffer_offset':
        boundary_=Polygon(boundary_proj.exterior.coords)
        LR_buffer=boundary_.buffer(buffer_distance,join_style=1).difference(boundary_)  #LinearRing  
        # LR_buffer=Polygon(boundary_proj.exterior.coords)
        LR_area=[LR_buffer.area]
        gpd.GeoDataFrame({'area': LR_area,'geometry':LR_buffer},crs=utm).to_crs(wgs84).to_file('./data/GIS/LR_buffer.shp')  
        return LR_buffer

def kml2gdf(fn,epsg=None,boundary=None): 
    import pandas as pd
    import geopandas as gpd
    
    # Enable fiona driver
    gpd.io.file.fiona.drvsupport.supported_drivers['KML'] = 'rw'
    kml_gdf=gpd.GeoDataFrame()
    for layer in tqdm(fiona.listlayers(fn)):
        src=fiona.open(fn, layer=layer)
        meta = src.meta
        meta['driver'] = 'KML'        
        with io.BytesIO() as buffer:
            with fiona.open(buffer, 'w', **meta) as dst:            
                for i, feature in enumerate(src):
                    if len(feature['geometry']['coordinates']) > 1:
                        # print(feature['geometry']['coordinates'])
                        dst.write(feature)
                        # break
            buffer.seek(0)
            one_layer=gpd.read_file(buffer,driver='KML')
            one_layer['group']=layer
            kml_gdf=kml_gdf.append(one_layer,ignore_index=True)            
    
    # crs={'init': 'epsg:4326'}
    if epsg is not None:
        kml_gdf_proj=kml_gdf.to_crs(epsg=epsg)

    if boundary:
        kml_gdf_proj['mask']=kml_gdf_proj.geometry.apply(lambda row:row.within(boundary))
        kml_gdf_proj.query('mask',inplace=True)        

    return kml_gdf_proj
    
def  shp2gdf(fn,epsg=None,boundary=None,encoding='utf-8'):
    import geopandas as gpd
    
    shp_gdf=gpd.read_file(fn,encoding=encoding)
    print('original data info:{}'.format(shp_gdf.shape))
    shp_gdf.dropna(how='all',axis=1,inplace=True)
    print('dropna-how=all,result:{}'.format(shp_gdf.shape))
    shp_gdf.dropna(inplace=True)
    print('dropna-several rows,result:{}'.format(shp_gdf.shape))
    # print(shp_gdf)
    if epsg is not None:
        shp_gdf_proj=shp_gdf.to_crs(epsg=epsg)
    if boundary:
        shp_gdf_proj['mask']=shp_gdf_proj.geometry.apply(lambda row:row.within(boundary))
        shp_gdf_proj.query('mask',inplace=True)        
    
    return shp_gdf_proj
      
def csv2gdf_A_taxi(data_root,epsg=None,boundary=None,): #
    import glob
    from pathlib import Path
    import geopandas as gpd
    import pandas as pd
    import datetime
    # from functools import reduce
    from tqdm import tqdm
    
    suffix='csv'
    fns=glob.glob(data_root+"/*.{}".format(suffix))
    fns_stem_df=pd.DataFrame([Path(fn).stem.split('_')[:2]+[fn] for fn in fns],columns=['info','date','file_path']).set_index(['info','date'])
    g_df_dict={}
    # i=0
    for info,g in tqdm(fns_stem_df.groupby(level=0)):
        g_df=pd.concat([pd.read_csv(fn).assign(date=idx[1]) for fn in g.file_path for idx in g.index]).rename({'value':'value_{}'.format(g.index[0][0])},axis=1) 
        g_df['time']=g_df.apply(lambda row:datetime.datetime.strptime(row.date+' {}:0:0'.format(row.hour), '%Y.%m.%d %H:%S:%f'),axis=1)
        g_gdf=gpd.GeoDataFrame(g_df,geometry=gpd.points_from_xy(g_df.longitude,g_df.latitude,),crs='epsg:4326')
        # print(g_gdf)
        if epsg is not None:
            g_gdf_proj=g_gdf.to_crs(epsg=epsg)
        if boundary:
            g_gdf_proj['mask']=g_gdf_proj.geometry.apply(lambda row:row.within(boundary))
            g_gdf_proj.query('mask',inplace=True)    
        
        g_df_dict['value_{}'.format(g.index[0][0])]=g_gdf_proj
        
        # if i==1:
        #     break
        # i+=1    
    return g_df_dict

def csv2gdf_A_POI(fn,epsg=None,boundary=None,encoding='utf-8'): #
    import glob
    from pathlib import Path
    import geopandas as gpd
    import pandas as pd
    from tqdm import tqdm
    
    csv_df=pd.read_csv(fn,encoding=encoding)
    csv_df['superclass']=csv_df['POI类型'].apply(lambda row:row.split(';')[0])
    # print(csv_df)
    # print(csv_df.columns)
    csv_gdf=gpd.GeoDataFrame(csv_df,geometry=gpd.points_from_xy(csv_df['经度'],csv_df['纬度']),crs='epsg:4326')

    if epsg is not None:
        csv_gdf_proj=csv_gdf.to_crs(epsg=epsg)
    if boundary:
        csv_gdf_proj['mask']=csv_gdf_proj.geometry.apply(lambda row:row.within(boundary))
        csv_gdf_proj.query('mask',inplace=True)  
    
    return csv_gdf_proj

def db2df(database_sql,table):
    import sqlite3
    import pandas as pd
    
    conn=sqlite3.connect(database_sql)
    df=pd.read_sql_query("SELECT * from {}".format(table), conn)
    # print(df)
    
    return df
    
def xls2gdf(fn,epsg=None,boundary=None,sheet_name=0):
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import LineString
    
    xls_df=pd.read_excel(fn,sheet_name=sheet_name)
    # print(xls_df)
    # print(xls_df.columns)
    xls_df['route_line']=xls_df.apply(lambda row:LineString([(row['开始维度'],row['开始经度'],),(row['结束维度'],row['结束经度'],)]),axis=1)
    xls_gdf=gpd.GeoDataFrame(xls_df,geometry=xls_df.route_line,crs='epsg:4326')    
    
    # print(xls_df)
    if epsg is not None:
        xls_gdf_proj=xls_gdf.to_crs(epsg=epsg)
    if boundary:
        xls_gdf_proj['mask']=xls_gdf_proj.geometry.apply(lambda row:row.within(boundary))
        xls_gdf_proj.query('mask',inplace=True)     
    
    return xls_gdf_proj


def Sentinel2_bandFNs(MTD_MSIL2A_fn):
    import xml.etree.ElementTree as ET
    '''
    funciton - 获取sentinel-2波段文件路径，和打印主要信息
    
    Paras:
    MTD_MSIL2A_fn - MTD_MSIL2A 文件路径
    
    Returns:
    band_fns_list - 波段相对路径列表
    band_fns_dict - 波段路径为值，反应波段信息的字段为键的字典
    '''
    Sentinel2_tree=ET.parse(MTD_MSIL2A_fn)
    Sentinel2_root=Sentinel2_tree.getroot()

    print("GENERATION_TIME:{}\nPRODUCT_TYPE:{}\nPROCESSING_LEVEL:{}".format(Sentinel2_root[0][0].find('GENERATION_TIME').text,
                                                           Sentinel2_root[0][0].find('PRODUCT_TYPE').text,                 
                                                           Sentinel2_root[0][0].find('PROCESSING_LEVEL').text
                                                          ))
    
    # print("MTD_MSIL2A.xml 文件父结构:")
    for child in Sentinel2_root:
        print(child.tag,"-",child.attrib)
    print("_"*50)    
    band_fns_list=[elem.text for elem in Sentinel2_root.iter('IMAGE_FILE')] #[elem.text for elem in Sentinel2_root[0][0][11][0][0].iter()]
    band_fns_dict={f.split('_')[-2]+'_'+f.split('_')[-1]:f+'.jp2' for f in band_fns_list}
    # print('get sentinel-2 bands path:\n',band_fns_dict)
    
    return band_fns_list,band_fns_dict  

# Function to normalize the grid values
def normalize_(array):
    """
    function - 数组标准化 Normalizes numpy arrays into scale 0.0 - 1.0
    """
    array_min, array_max = array.min(), array.max()
    return ((array - array_min)/(array_max - array_min))

def sentinel_2_NDVI(sentinel_2_root,save_path):
    import os
    import earthpy.spatial as es
    import rasterio as rio
    from tqdm import tqdm
    import shapely
    import numpy as np
    from scipy import stats
    from osgeo import gdal
    
    MTD_fn=os.path.join(sentinel_2_root,'MTD_MSIL2A.xml')
    band_fns_list,band_fns_dict=Sentinel2_bandFNs(MTD_fn) 
    # print(band_fns_dict).
    bands_selection=["B02_10m","B03_10m","B04_10m","B08_10m"] 
    stack_bands=[os.path.join(sentinel_2_root,band_fns_dict[b]) for b in bands_selection]
    # print(stack_bands)
    array_stack, meta_data=es.stack(stack_bands)    
    meta_data.update(
        count=1,
        dtype=rio.float64,
        driver='GTiff'
        )   
    print("meta_data:\n",meta_data)   

    NDVI=(array_stack[3]-array_stack[2])/(array_stack[3]+array_stack[2])    
    with rio.open(save_path,'w',**meta_data) as dst:
        dst.write(np.expand_dims(NDVI.astype(meta_data['dtype']),axis=0))
    print('NDVI has been saved as raster .tif format....')
    
    return NDVI

def raster_crop(raster_fn,crop_shp_fn,boundary=None):
    import rasterio as rio
    import geopandas as gpd
    import earthpy.spatial as es
    import numpy as np
    import earthpy.plot as ep
    from shapely.geometry import shape
    
    with rio.open(raster_fn) as src:
        ori_raster=src.read(1)
        ori_profile=src.profile
    print(ori_raster.shape)
    crop_boundary=gpd.read_file(crop_shp_fn).to_crs(ori_profile['crs'])
    # print(crop_boundary)
    print("_"*50)
    print(' crop_boundary: {}'.format(crop_boundary.crs))
    print("_"*50)
    print(' ori_raster: {}'.format( ori_profile['crs']))
    
    with rio.open(raster_fn) as src:
        cropped_img, cropped_meta=es.crop_image(src,crop_boundary)
    print(cropped_img.shape)
    
    cropped_meta.update({"driver": "GTiff",
                         "height": cropped_img.shape[0],
                         "width":  cropped_img.shape[1],
                         "transform": cropped_meta["transform"]})
    cropped_img_mask=np.ma.masked_equal(cropped_img[0], -9999.0) 
    # print(cropped_img_mask)
    ep.plot_bands(cropped_img_mask, cmap='terrain', cbar=False) 
    print(type(cropped_img_mask))
    
    cropped_shapes=(
        {'properties': {'raster_val': v}, 'geometry': s}
        for i, (s, v) in enumerate(rio.features.shapes(cropped_img.astype(np.float32),transform=cropped_meta['transform']))) #,mask=None
    # print(cropped_shapes)
    geoms=list(cropped_shapes)    
    print(geoms[0])
    cropped_img_gpd=gpd.GeoDataFrame.from_features(geoms)
    cropped_img_gpd.geometry=cropped_img_gpd.geometry.apply(lambda row:row.centroid)
    # print(cropped_img_gpd)

    if boundary:
        cropped_img_gpd['mask']=cropped_img_gpd.geometry.apply(lambda row:row.within(boundary))
        cropped_img_gpd.query('mask',inplace=True)      
    
    return cropped_img_gpd

def gpd2SQLite(gdf_,db_fp,table_name):
    from geoalchemy2 import Geometry, WKTElement
    from sqlalchemy import create_engine
    import pandas as pd    
    import copy
    import shapely.wkb    

    gdf=copy.deepcopy(gdf_)
    crs=gdf.crs
    # print(help(crs))
    # print(crs.to_epsg())
    # gdf['geom']=gdf['geometry'].apply(lambda g: WKTElement(g.wkt,srid=crs.to_epsg()))
    #convert all values from the geopandas geometry column into their well-known-binary representations
    gdf['geom']=gdf.apply(lambda row: shapely.wkb.dumps(row.geometry),axis=1)
    gdf.drop(columns=['geometry','mask'],inplace=True)
    # print(type(gdf.geom.iloc[0]))
    print(gdf)
    engine=create_engine('sqlite:///'+'\\\\'.join(db_fp.split('\\')),echo=True)
    gdf.to_sql(table_name, con=engine, if_exists='replace', index=False,) #dtype={'geometry': Geometry('POINT')} ;dtype={'geometry': Geometry('POINT',srid=crs.to_epsg())}
    print('has been written to into the SQLite database...')

def gpd2postSQL(gdf,table_name,**kwargs):
    from sqlalchemy import create_engine
    # engine=create_engine("postgres://postgres:123456@localhost:5432/workshop-LA-UP_IIT")  
    engine=create_engine("postgres://{myusername}:{mypassword}@localhost:5432/{mydatabase}".format(myusername=kwargs['myusername'],mypassword=kwargs['mypassword'],mydatabase=kwargs['mydatabase']))  
    gdf.to_postgis(table_name, con=engine, if_exists='replace', index=False,)  
    print("_"*50)
    print('has been written to into the PostSQL database...')
    
def postSQL2gpd(table_name,geom_col='geometry',**kwargs):
    from sqlalchemy import create_engine
    import geopandas as gpd
    engine=create_engine("postgres://{myusername}:{mypassword}@localhost:5432/{mydatabase}".format(myusername=kwargs['myusername'],mypassword=kwargs['mypassword'],mydatabase=kwargs['mydatabase']))  
    gdf=gpd.read_postgis(table_name, con=engine,geom_col=geom_col)
    print("_"*50)
    print('The data has been read from PostSQL database...')    
    return gdf

def raster2postSQL(raster_fn,**kwargs):
    from osgeo import gdal, osr
    import psycopg2
    import subprocess
    from pathlib import Path
    
    raster=gdal.Open(raster_fn)
    # print(raster)
    proj=osr.SpatialReference(wkt=raster.GetProjection())
    print(proj)
    projection=str(proj.GetAttrValue('AUTHORITY',1))
    gt=raster.GetGeoTransform()
    pixelSizeX=str(round(gt[1]))
    pixelSizeY=str(round(-gt[5]))
    
    # cmds='raster2pgsql -s '+projection+' -I -C -M "'+raster_fn+'" -F -t '+pixelSizeX+'x'+pixelSizeY+' public.'+'uu'+' | psql -d {mydatabase} -U {myusername} -h localhost -p 5432'.format(mydatabase=kwargs['mydatabase'],myusername=kwargs['myusername'])
    cmds='raster2pgsql -s '+projection+' -I -M "'+raster_fn+'" -F -t '+pixelSizeX+'x'+pixelSizeY+' public.'+Path(raster_fn).stem+' | psql -d {mydatabase} -U {myusername} -h localhost -p 5432'.format(mydatabase=kwargs['mydatabase'],myusername=kwargs['myusername'])
    print("_"*50)
    print(cmds)
    subprocess.call(cmds, shell=True)
    print("_"*50)
    print('The raster has been loaded into PostSQL...')
    
    

if __name__=="__main__":
    #a-create or connect to the database
    db_file=r'./database/workshop_LAUP_iit.db'
    # sql_w=SQLite_handle(db_file)
    # sql_w.create_connection()    
    
    #b-create boundary
    #method_a
    kml_extent=data_dic['qingliangMountain_boundary']
    boudnary_polygon=boundary_buffer_centroidCircle(kml_extent,nanjing_epsg,bounadry_type='buffer_circle',buffer_distance=5000) #'buffer_circle';'buffer_offset'
        
    #c-road_network_kml
    # road_gdf=kml2gdf(data_dic['road_network'],epsg=nanjing_epsg,boundary=boudnary_polygon)
    # road_gdf.plot()
    # gpd2postSQL(road_gdf,table_name='road_network',myusername='postgres',mypassword='123456',mydatabase='workshop-LA-UP_IIT')
    
    #d-02_building footprint
    # buildingFootprint=shp2gdf(data_dic['building_footprint'],epsg=nanjing_epsg,boundary=boudnary_polygon)
    # buildingFootprint.plot(column='Floor',cmap='terrain')
    # gpd2postSQL(buildingFootprint,table_name='building_footprint',myusername='postgres',mypassword='123456',mydatabase='workshop-LA-UP_IIT')
    
    # d-03_bus routes
    # bus_routes=shp2gdf(data_dic['bus_routes'],epsg=nanjing_epsg,boundary=None,encoding='GBK')
    # bus_routes.plot()
    # gpd2postSQL(bus_routes,table_name='bus_routes',myusername='postgres',mypassword='123456',mydatabase='workshop-LA-UP_IIT')
    
    #d-04_bus station
    # bus_stations=shp2gdf(data_dic['bus_stations'],epsg=nanjing_epsg,boundary=boudnary_polygon,encoding='GBK')
    # bus_stations.plot()
    # gpd2postSQL(bus_stations,table_name='bus_stations',myusername='postgres',mypassword='123456',mydatabase='workshop-LA-UP_IIT')
    
    #d-05_subway lines
    # subway_lines=shp2gdf(data_dic['subway_lines'],epsg=nanjing_epsg,encoding='GBK')
    # subway_lines.plot()
    # gpd2postSQL(subway_lines,table_name='subway_lines',myusername='postgres',mypassword='123456',mydatabase='workshop-LA-UP_IIT')
    
    #d-06_subway stations
    # subway_stations=shp2gdf(data_dic['subway_stations'],epsg=nanjing_epsg,boundary=boudnary_polygon,encoding='GBK')
    # subway_stations.plot()
    # gpd2postSQL(subway_stations,table_name='subway_stations',myusername='postgres',mypassword='123456',mydatabase='workshop-LA-UP_IIT')
    
    #d-07_population
    # population=shp2gdf(data_dic['population'],epsg=nanjing_epsg,boundary=boudnary_polygon,encoding='GBK')
    # population.plot(column='Population',cmap='hot')
    # gpd2postSQL(population,table_name='population',myusername='postgres',mypassword='123456',mydatabase='workshop-LA-UP_IIT')
    
    #e-A-08_Nanjing taxi data
    # g_df_dict=csv2gdf_A_taxi(data_dic['taxi'],epsg=nanjing_epsg,boundary=boudnary_polygon)
    # taxi_keys=list(g_df_dict.keys())
    # print(taxi_keys)
    # g_df_dict[taxi_keys[0]].plot(column=taxi_keys[0],cmap='hot')
    
    # for key in taxi_keys:
    #     gpd2postSQL(g_df_dict[key],table_name='taxi_{}'.format(key),myusername='postgres',mypassword='123456',mydatabase='workshop-LA-UP_IIT') 

    #e-B-09_POI
    # POI=csv2gdf_A_POI(data_dic['POI'],epsg=nanjing_epsg,boundary=boudnary_polygon,encoding='GBK')
    # POI.plot(column='superclass',cmap='terrain',markersize=1)
    
    # POI.rename({'唯一ID':'ID', 
    #             'POI名称':"Name", 
    #             'POI类型':"class", 
    #             'POI类型编号':"class_idx", 
    #             '行业类型':"industry_class", 
    #             '地址':"address", 
    #             '经度':"lon", 
    #             '纬度':'lat',
    #             'POI所在省份名称':"province", 
    #             'POI所在城市名称':"city", 
    #             '区域编码':"reginal_code", 
    #             # 'superclass', 
    #             # 'geometry', 
    #             # 'mask'
    #             },axis=1,inplace=True)
    # #table name should be the low case
    # gpd2postSQL(POI,table_name='poi',myusername='postgres',mypassword='123456',mydatabase='workshop-LA-UP_IIT')

    # f-10_ Metro Weibo(microblog) publish
    # microblog=db2df(data_dic['microblog'],'NajingMetro')
    
    #g-11_bike sharing/ no data were available for Nanjing area
    # bike_sharing=xls2gdf(data_dic['bike_sharing'],epsg=nanjing_epsg,boundary=boudnary_polygon,sheet_name='共享单车数据a')
    # bike_sharing.plot()
    
    #h-12_sentinel-2-NDVI
    # ndvi_fn=r'C:\Users\richi\omen_richiebao\omen_IIIT\workshop_LA_UP_iit\data\RS\NDVI.tif'
    # sentinel_2_NDVI=sentinel_2_NDVI(data_dic['sentinel_2'],ndvi_fn)
    # #i-12-01_raster crop
    # ndvi_cropped=raster_crop(raster_fn=ndvi_fn,crop_shp_fn='./data/GIS/b_centroid_buffer.shp',boundary=boudnary_polygon) #,cropped_fn='./data/GIS/NDVI_cropped.tif'
    # ndvi_cropped.plot(column='raster_val',cmap='terrain',markersize=1)
    # gpd2postSQL(ndvi_cropped,table_name='ndvi',myusername='postgres',mypassword='123456',mydatabase='workshop-LA-UP_IIT')
       
    
    #I-write GeoDataFrame into SQLite database
    # gpd2SQLite(population,db_file,table_name='population')
        
    #G-write GeoDataFrame into PostgreSQL 
    # gpd2postSQL(population,table_name='population',myusername='postgres',mypassword='123456',mydatabase='workshop-LA-UP_IIT')
    #G-read GeoDataFrame from PostgreSQL
    # population_postsql=postSQL2gpd(table_name='population',geom_col='geometry',myusername='postgres',mypassword='123456',mydatabase='workshop-LA-UP_IIT')
    # population_postsql.plot(column='Population',cmap='hot')
    
    #H-load raster into postGreSQL
    # raster2postSQL(ndvi_fn,table_name='ndvi',myusername='postgres',mypassword='123456',mydatabase='workshop-LA-UP_IIT')
        
