import pickle
import geohash
import pandas as pd
import numpy as np
from polygon_geohasher.polygon_geohasher import geohash_to_polygon
# import folium
# from polygon_geohasher import geohash_to_polygon
import geojson as json
import numpy as np
# import geopandas as gpd
import pandas as pd
import torch
import collections

from model.args import Args

_base32 = '0123456789bcdefghjkmnpqrstuvwxyz'
_base32_map = {}
for i in range(len(_base32)):
    _base32_map[_base32[i]] = i
del i
import os





class function :
    args = Args()
    #**********************************************************************************

    def Weather_cleaning(self,Weather):
        Weather['Time'] = pd.to_datetime(Weather['Time'])
        Weather['Time'] = Weather['Time'].dt.strftime('%Y-%m-%d %H:00:00')
        Data_Weather = pd.pivot_table(Weather, values=['T', 'U', 'Ff'], index=['Time'], aggfunc=np.mean)
        Data_Weather['Time'] = Data_Weather.index
        Data_Weather['Time'] = pd.to_datetime(Data_Weather['Time'])
        # Data_Weather['day_of_week'] = Data_Weather['Time'].dt.day_name()
        Data_Weather['hour_of_day'] = Data_Weather['Time'].dt.hour
        Data_Weather = Data_Weather.drop(['Time'], axis=1)
        dfDummies = pd.get_dummies(Data_Weather['hour_of_day'], prefix='HOUR')

        for i in range(0, 24):
            col = "HOUR_" + str(i)
            Data_Weather[col]=0
        Data_Weather[dfDummies.columns]=dfDummies
        # dfDummies = pd.get_dummies(Data_Weather['day_of_week'], prefix = 'category')
        Data_Weather = Data_Weather.drop(['hour_of_day'], axis=1)
        # Data_Weather=Data_Weather.drop(['day_of_week'],axis=1)
        # df=Data_Weather
        df = Data_Weather#pd.concat([Data_Weather, dfDummies], axis=1)
        # df=df.reindex(['T','U','Ff','hour_of_day','category_Sunday','category_Monday','category_Thursday','category_Tuesday','category_Wednesday','category_Saturday'],axis=1)
        df = df.rename(columns={"T": "temperature_of_the_day", "U": "Humidity", 'Ff': 'Wind_speed'})
        df = df.drop(["temperature_of_the_day", "Humidity"], axis=1)
        #df = df.loc[(df.index >= str(self.args.date_rng.min())) & (df.index <= str(self.args.date_rng.max()))]
        #df.index = pd.DatetimeIndex(df.index)
        #df = df.reindex(self.args.date_rng, fill_value=0)
        df = df.fillna(0)

        return df
    #**********************************************************************************
    def geohashFunction(self,x,y):
       return geohash.encode(x,y,6)
    #**********************************************************************************

    def decode_c2i(self,hashcode):
        lon = 0
        lat = 0
        bit_length = 0
        lat_length = 0
        lon_length = 0
        for i in hashcode:
            t = _base32_map[i]
            if bit_length % 2 == 0:
                lon = lon << 3
                lat = lat << 2
                lon += (t >> 2) & 4
                lat += (t >> 2) & 2
                lon += (t >> 1) & 2
                lat += (t >> 1) & 1
                lon += t & 1
                lon_length += 3
                lat_length += 2
            else:
                lon = lon << 2
                lat = lat << 3
                lat += (t >> 2) & 4
                lon += (t >> 2) & 2
                lat += (t >> 1) & 2
                lon += (t >> 1) & 1
                lat += t & 1
                lon_length += 2
                lat_length += 3

            bit_length += 5

        return (lat, lon, lat_length, lon_length)

    # **********************************************************************************
    def encode_i2c(self,lat, lon, lat_length, lon_length):
        precision = int((lat_length + lon_length) / 5)
        if lat_length < lon_length:
            a = lon
            b = lat
        else:
            a = lat
            b = lon

        boost = (0, 1, 4, 5, 16, 17, 20, 21)
        ret = ''
        for i in range(precision):
            ret += _base32[(boost[a & 7] + (boost[b & 3] << 1)) & 0x1F]
            t = a >> 3
            a = b >> 2
            b = t

        return ret[::-1]

    # **********************************************************************************
    def neighbors(self,hashcode, S):
        (lat, lon, lat_length, lon_length) = self.decode_c2i(hashcode)
        ret = []
        rang = S // 2
        tab = []
        tab2 = []
        tab3 = []

        for i in range(1, rang + 1):
            tab.append(lat + rang + 1 - i)
            tab3.append(lat - i)
        for i in range(-rang, rang + 1):
            tab2.append(lon + i)

        for tlat in tab:
            if not tlat >> lat_length:
                for tlon in tab2:
                    ret.append(self.encode_i2c(tlat, tlon, lat_length, lon_length))

        tlat = lat
        for tlon in tab2:
            code = self.encode_i2c(tlat, tlon, lat_length, lon_length)
            if code:
                ret.append(code)

        for tlat in tab3:
            if tlat >= 0:
                for tlon in tab2:
                    ret.append(self.encode_i2c(tlat, tlon, lat_length, lon_length))

        return np.array(ret).reshape(S, S)

    # **********************************************************************************
    def clean_data(self,df) :
      #data=df.loc[(df['requested_date'] >= str(self.args.date_rng.min())) & (df['requested_date']<= str(self.args.date_rng.max()))]
      data=df[['p_lat','p_lng','requested_date']]
      data['geohash']=data.apply(lambda x: self.geohashFunction(x.p_lat, x.p_lng), axis=1)
      data['geometry']=data['geohash'].apply(geohash_to_polygon)
      data['requested_date']=pd.to_datetime(data['requested_date'])
      data['requested_date']=data['requested_date'].dt.strftime('%Y-%m-%d %H:00:00')
      return data

    # **********************************************************************************
    def seq_of_demand_zone(self,data,zone,reshape) :
       x=[]
       Data=pd.crosstab(data['requested_date'],data['geohash'])
       Data.index = pd.DatetimeIndex(Data.index)
       #Data= Data.reindex(self.args.date_rng, fill_value=0)
       for zn in self.neighbors(zone,self.args.img_size).reshape(self.args.img_size*self.args.img_size,)  :
         if zn not in Data.columns:
            #print(zn)
            Data[str(zn)]= 0
       Data=Data[self.neighbors(zone,self.args.img_size).reshape(self.args.img_size*self.args.img_size,)]
       Data= Data.reindex(self.neighbors(zone,self.args.img_size).reshape(self.args.img_size*self.args.img_size,), axis=1)
       for i in Data.index:
            x.append(torch.from_numpy(np.array(Data.loc[i]).reshape(self.args.img_size,self.args.img_size)))
       tensor= torch.stack(x)
       #print(tensor.reshape(reshape,1,self.args.img_size,self.args.img_size).shape)
       return tensor.reshape(reshape,1,self.args.img_size,self.args.img_size)

    # **********************************************************************************
    def create_inout_sequences(self,input_data, tw):
        inout_seq = []
        L = len(input_data)
        for i in range(L-tw,L-tw+1):
            train_seq = input_data[i:i+tw]
            #train_label = input_data[i+tw:i+tw+1]
            inout_seq.append(train_seq )
        return inout_seq

    # **********************************************************************************
    def create_inout_sequences_extarnal_data(self,input_data, tw):
        inout_seq_external_data = []
        L = len(input_data)
        for i in range(L-tw,L-tw+1):
            train_seq_external_data = input_data[i:i+tw]
            train_label = input_data[i+tw:i+tw+1]
            inout_seq_external_data.append(train_seq_external_data.values)
        return torch.FloatTensor(inout_seq_external_data).view(1 ,self.args.seq_len,len(input_data.columns))

    # **********************************************************************************
    def create_data_final(self,data,number_of_zone,sequence_len):
      sequence_data=[]
      Data=pd.crosstab(data['requested_date'],data['geohash'])
      Data.index = pd.DatetimeIndex(Data.index)
      Data=Data.reindex(Data.mean().sort_values(ascending=False).index, axis=1).iloc[:,0:self.args.number_of_zone_training]
      for zone in Data.columns:
        #print(zone)
        sequence_data.append(self.create_inout_sequences(self.seq_of_demand_zone(data,zone,8),sequence_len))
      return sequence_data