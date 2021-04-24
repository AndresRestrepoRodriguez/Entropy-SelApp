import pandas as pd
import numpy as np
from scipy.io import arff as ar


def read_data(ruta, file_extension):
    if file_extension == "csv":
        return read_csv(ruta)
    elif file_extension == "arff":
        return read_arff(ruta)
    else:
        return None, None, None, None


def read_arff(ruta):
    data_transform_temp = []
    data_transform = []
    data, meta = ar.loadarff(ruta)
    columns = meta.names()
    typescol = meta.types()
    for i in data:
        data_transform_temp = [x.decode('UTF-8') if type(x) == np.bytes_ else x for x in i]
        data_transform.append(data_transform_temp)
        data_transform_temp = []
    dataframe = pd.DataFrame(data_transform, columns=columns)
    dataframe.to_csv('./static/data/datos.csv', index=False)
    counter_data = dataframe.count().values.tolist()
    criterion = generate_criterion(typescol)
    return columns, counter_data, typescol, criterion


def generate_criterion(attributes_types):
    set_attributes = set(attributes_types)
    result = len(set_attributes) == 1
    type_set = set_attributes.pop()
    if result and type_set=='numeric':
        criterion = 'Euclidean'
    elif result and type_set== 'nominal':
        criterion = 'Hamming'
    else:
        criterion = 'Chimerge'
    return criterion


def read_csv_dataframe(ruta):
    return pd.read_csv(ruta)


def read_csv(ruta):
    dataframe = pd.read_csv(ruta)
    dataframe.to_csv('./static/data/datos.csv', index=False)
    counter_data = dataframe.count().values.tolist()
    columns = dataframe.columns.tolist()
    typescol = get_types(dataframe)
    criterion = generate_criterion(typescol)
    return columns, counter_data, typescol, criterion


def get_types(dataframe):
    types_list = []
    for i in dataframe.columns:
        if dataframe[i].dtype == np.float64 or dataframe[i].dtype == np.int64:
            types_list.append('numeric')
        else:
            types_list.append('nominal')
    return types_list
