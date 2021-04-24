from scipy.io import arff
import pandas as pd
import numpy as np
import math


def calculate_matrix_euclidean(data):
    alpha = 0.5
    maxFeatures = data.max()
    minFeatures = data.min()
    diffFeatures = (maxFeatures-minFeatures).values
    indexDf = data.index.values
    dfTestValues = data.values
    matrixRes = np.zeros((len(indexDf),len(indexDf)))
    for i in range(len(indexDf)):
        for j in range(i+1,(len(indexDf))):
            array_temp = []
            for k in range(len(dfTestValues[i])):
                value_temp = pow((dfTestValues[i][k]-dfTestValues[j][k])/diffFeatures[k],2)
                array_temp.append(value_temp)
            value_Dij = sum(array_temp)
            value_Dij = math.sqrt(value_Dij)
            valor_matij = pow(math.e, ((-1)*alpha*value_Dij))
            matrixRes[i][j] = valor_matij
    return matrixRes


def calculate_matrix_hamming(data):
    len_columns = len(data.columns)
    indexDf = data.index.values
    dfTestValues = data.values
    matrixRes = np.zeros((len(indexDf), len(indexDf)))
    for i in range(len(indexDf)):
        for j in range(i+1,(len(indexDf))):
            array_temp = []
            for k in range(len(dfTestValues[i])):
                if dfTestValues[i][k] == dfTestValues[j][k]:
                    value_temp = 1
                else:
                    value_temp = 0
                array_temp.append(value_temp)
            valor_matij = sum(array_temp)/len_columns
            matrixRes[i][j] = valor_matij
    return matrixRes


def calculate_entropy(matrix):
    matrix_local = matrix.copy()
    for i in range(len(matrix_local)):
        for j in range(len(matrix_local)):
            if matrix_local[i][j] != 0 and matrix_local[i][j] != 1:
                valor_temp = (matrix_local[i][j] * math.log2(matrix_local[i][j]))\
                             + ((1-matrix_local[i][j]) * math.log2((1-matrix_local[i][j])))
                matrix_local[i][j] = (-1)*valor_temp
            elif matrix_local[i][j] == 1:
                matrix_local[i][j] = 0
    array_temp_entropy = []
    for i in range(len(matrix_local)):
        valor_temp = sum(matrix_local[i])
        array_temp_entropy.append(valor_temp)
    entropy = sum(array_temp_entropy)
    return entropy


def suggestion_generation(results_diff, entropies_results):
    attributes_array = []
    min_diff = min(results_diff)
    for i in range(len(entropies_results)):
        if entropies_results[i][2] == min_diff:
            attributes_array.append(entropies_results[i][0][4:])
    return attributes_array


def feature_selection_euclidean(data):
    array_process = []
    array_temp = []
    array_results = []
    array_diff = []
    columns = data.columns.values
    general_result = calculate_matrix_euclidean(data)
    array_process.append('Cálculo de matriz de distancia Euclideana con la totalidad de datos')
    general_entropy = calculate_entropy(general_result)
    array_process.append('Cálculo de entropía con la totalidad de datos')
    array_temp.append('General')
    array_temp.append(round(general_entropy, 2))
    array_temp.append(round((general_entropy-general_entropy), 2))
    array_results.append(array_temp)
    for i in range(len(columns)):
        array_temp = []
        df_aux = data.drop([columns[i]], axis=1)
        particular_result = calculate_matrix_euclidean(df_aux)
        array_process.append(f'Cálculo de matriz de distancia Euclideana sin el atributo {columns[i]}')
        particular_entropy = calculate_entropy(particular_result)
        diff_entropy = abs(general_entropy - particular_entropy)
        array_process.append(f'Cálculo de entropía sin el atributo {columns[i]}')
        array_temp.append(f'Sin {columns[i]}')
        array_temp.append(round(particular_entropy,2))
        array_temp.append(round(diff_entropy,2))
        array_diff.append(round(diff_entropy,2))
        array_results.append(array_temp)
    array_suggestion = suggestion_generation(array_diff,array_results)
    return array_process, array_results, array_suggestion


def feature_selection_hamming(data):
    array_process = []
    array_temp = []
    array_results = []
    array_diff = []
    columns = data.columns.values
    general_result = calculate_matrix_hamming(data)
    array_process.append('Cálculo de matriz de distancia Hamming con la totalidad de datos')
    general_entropy = calculate_entropy(general_result)
    array_process.append('Cálculo de entropía con la totalidad de datos')
    array_temp.append('General')
    array_temp.append(round(general_entropy,2))
    array_temp.append(round((general_entropy-general_entropy),2))
    array_results.append(array_temp)
    for i in range(len(columns)):
        array_temp = []
        df_aux = data.drop([columns[i]], axis=1)
        particular_result = calculate_matrix_hamming(df_aux)
        array_process.append(f'Cálculo de matriz de distancia Hamming sin el atributo {columns[i]}')
        particular_entropy = calculate_entropy(particular_result)
        array_process.append(f'Cálculo de entropía sin el atributo {columns[i]}')
        diff_entropy = abs(general_entropy - particular_entropy)
        array_temp.append(f'Sin {columns[i]}')
        array_temp.append(round(particular_entropy,2))
        array_temp.append(round(diff_entropy,2))
        array_diff.append(round(diff_entropy,2))
        array_results.append(array_temp)
    array_suggestion = suggestion_generation(array_diff,array_results)
    return array_process, array_results, array_suggestion
