from model.model_class import RMSELoss
from model.model_class import CNN

from model.model_class import Combine
from model.functions import function
import numpy as np
from model.args import Args
from flask import Flask, request, render_template, jsonify
import pickle
import geohash
import pandas as pd
import numpy as np
# import folium
# from polygon_geohasher import geohash_to_polygon
import geojson as json
import numpy as np
# import geopandas as gpd
import pandas as pd
import collections
# import statisticsc
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# import torchvision
# from torchvision import datasets, transforms
from torch.optim.lr_scheduler import LambdaLR
from torch.autograd import Variable

_base32 = '0123456789bcdefghjkmnpqrstuvwxyz'
_base32_map = {}
for i in range(len(_base32)):
    _base32_map[_base32[i]] = i
del i
import os

# Initializing the FLASK API
app = Flask(__name__)

args = Args()
model = Combine(args.img_size, args.num_filtre, args.size_filtre, args.kernel_maxpooling, args.stride_maxpooling,
                args.output_size_linear, args.hidden_size, args.output_size_linear_lstm, args.batsh_size, args.seq_len)

model.load_state_dict(torch.load("C:/Users/pc/Desktop/RestAPI_Flask/model/model.pt"))
model = model.eval()
print(model)
fun = function()

@app.route('/')
def home():
    return "adem"

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        content = json.load(request.files['data'])
        Weather=json.load(request.files['weather'])

    data = fun.clean_data(pd.DataFrame(content))
    External_Feautre = fun.Weather_cleaning(pd.DataFrame(Weather))
    External_Feautre_sequence = fun.create_inout_sequences_extarnal_data(External_Feautre, args.seq_len)
    seq = fun.create_data_final(data, args.number_of_zone_training, args.seq_len)
    #print(seq[1][0].shape)
    prediction=[]
    for i in range(0,1):
        input_data = []
        y = []
        for j in range(args.number_of_zone_training):
            #print(seq[j][i][0])
            #print("****************")
            input_data.append(seq[j][0])
        x_train_external_data = External_Feautre_sequence[i]
        x_train_zones_seq = torch.cat(input_data)
        #print(x_train_zones_seq.shape)
        pred = model(x_train_zones_seq.float(), x_train_external_data.float())
        prediction.append(pred.detach().numpy().reshape(args.number_of_zone_training))
    print(pd.Series(prediction[0]).to_json(orient='values'))
    Data = pd.crosstab(data['requested_date'], data['geohash'])
    Data.index = pd.DatetimeIndex(Data.index)
    Data = Data.reindex(Data.mean().sort_values(ascending=False).index, axis=1).iloc[:, 0:args.number_of_zone_training]
    values=pd.Series(np.array(prediction[0]).round().astype(np.int))
    dataFolium = pd.DataFrame({'geohash': Data.columns, 'value': values})
    return  dataFolium.to_json(orient='values')

if __name__ == "__main__":


    app.run(debug=True)
