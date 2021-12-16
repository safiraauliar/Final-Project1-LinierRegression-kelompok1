import pickle
import json 
import numpy as np
import warnings
warnings.filterwarnings("ignore")

__cab_names = None
__data_columns = None
__model = None

def get_predicted_price(cab_name,distance,surge_multiplier): 
    try:   
        loc_index = __data_columns.index(cab_name.lower())
    except:
        loc_index = -1

    x = np.zeros(len(__data_columns))
    x[0] = distance
    x[1] = surge_multiplier
    if loc_index >= 0:
        x[loc_index] = 1

    return round(__model.predict([x])[0],2)


def load_saved_artifacts():
    print("loading saved artifacts...start")
    global __data_columns
    global __cab_names
    global __model

    with open("./artifacts/columns.json",'r') as f:
        __data_columns = json.load(f)['data_columns']
        __cab_names = __data_columns[2:]

    
    if __model is None:
        with open("./artifacts/predict_price_model.pkl",'rb') as f:
            __model = pickle.load(f)
    print("loading saved artifacts ... done")

def get_cab_names():
    return __cab_names

def get_data_columns():
    return __data_columns

if __name__=='__main__':
    load_saved_artifacts()
    print(get_cab_names())
    print(get_data_columns())
    print(get_predicted_price('Lyft Lux',0.44,0.0))
    print(get_predicted_price('Lyft',0.44,0.0))
    print(get_predicted_price('Lyft Lux Black XL',1.0, 0.0))
