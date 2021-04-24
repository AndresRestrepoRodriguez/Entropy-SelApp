# -*- coding: utf-8 -*-
import numpy as np
from scipy.stats import chi2
import math


def chimerge_discretization_individual(dataframe, numeric_column, labeled_attribute, confianza_parametro):
    data_sort = dataframe[[numeric_column, labeled_attribute]]
    data_sort.sort_values(by=[numeric_column], inplace=True)
    to_disc = data_sort[numeric_column].values
    labeled = data_sort[labeled_attribute].values
    min_to_disc = np.amin(to_disc)
    max_to_disc = np.amax(to_disc)
    diff_mm = max_to_disc-min_to_disc
    num_partitions = 6
    val_int = diff_mm/num_partitions
    array_initial_intervals = []
    increase = 0.01
    initial = min_to_disc
    for i in range(num_partitions):
        array_temp = list()
        array_temp.append(initial)
        next = round((initial + val_int), 2)
        array_temp.append(next)
        initial = round((next+increase), 2)
        array_initial_intervals.append(array_temp)

    classes_labeled = list(set(labeled))
    num_classes = len(classes_labeled)
    degree_freedom = num_classes-1
    confidence = confianza_parametro
    threshold = chi2.ppf(confidence, degree_freedom)

    final_intervals = []
    while True:
        mat_frecuency = np.empty([2, num_classes])
        for i in range(2):
            for j in range(num_classes):
                df_aux = data_sort.loc[(data_sort[numeric_column] >= array_initial_intervals[i][0])
                                       & (data_sort[numeric_column] <= array_initial_intervals[i][1])]
                df_aux_labeled = df_aux.loc[(df_aux[labeled_attribute] == classes_labeled[j])]
                mat_frecuency[i][j] = len(df_aux_labeled)
            r_sum = np.sum(mat_frecuency, axis=1)
            c_cum = np.sum(mat_frecuency, axis=0)
            n_value = np.sum(c_cum)

            array_X = []
            for k in range(mat_frecuency.shape[0]):
                for m in range(mat_frecuency.shape[1]):
                    e_temp = (r_sum[k]*c_cum[m])/n_value
                    if e_temp == 0:
                        e_temp = 0.1
                    valor_temp = (math.pow(((mat_frecuency[k][m])-e_temp),2))/e_temp
                    array_X.append(valor_temp)
                X_final = sum(array_X)
        if X_final <= threshold:
            int_unified = [array_initial_intervals[i-1][0], array_initial_intervals[i][1]]
            array_initial_intervals_temp = list()
            array_initial_intervals_temp.append(int_unified)
            array_initial_intervals_temp.extend(array_initial_intervals[i+1:])
            array_initial_intervals = array_initial_intervals_temp.copy()
            if len(array_initial_intervals)==1:
                final_intervals.append(array_initial_intervals[0])
                return final_intervals
        else:
            final_intervals.append(array_initial_intervals[i-1])
            del array_initial_intervals[i-1]
            if len(array_initial_intervals) == 1:
                final_intervals.append(array_initial_intervals[0])
                return final_intervals
            if len(array_initial_intervals) == 0:
                return final_intervals


def replace_discretization(intervals, numeric_column, dataframe):
    data_col = dataframe[numeric_column].values
    data_new_col = np.zeros_like(data_col).astype(np.str)
    for i in range(len(data_col)):
        for k in range(len(intervals)):
            if intervals[k][0] <= data_col[i] <= intervals[k][1]:
                data_new_col[i] = str(intervals[k])
    dataframe[numeric_column] = data_new_col


def get_numeric_columns(name_cols, type_cols):
    array_numeric_cols = []
    for i in range(len(type_cols)):
        if type_cols[i] == 'numeric':
            array_numeric_cols.append(name_cols[i])
    return array_numeric_cols


def chimerge_general(dataframe, numeric_columns, labeled_attribute, confidence_p):
    array_process = []
    for k in range(len(numeric_columns)):
        array_process.append('Discretización del atributo: '+numeric_columns[k])
        intervals = chimerge_discretization_individual(dataframe, numeric_columns[k], labeled_attribute, confidence_p)
        array_process.append('Intervalos obtenidos: '+str(intervals))
        replace_discretization(intervals, numeric_columns[k], dataframe)
    return dataframe, array_process
