from flask import Flask, render_template, request, jsonify, redirect
from werkzeug.utils import secure_filename
import os
import dataoverview as do
import featureselection as fs
import chimerge as ch


UPLOAD_FOLDER = './static/data/'

app = Flask(__name__)

app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = set(['arff', 'csv'])

data_path = ''
csv_data_path = './static/data/datos.csv'
columns_name = []
types_data = []


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_extension(filename):
    return filename.rsplit('.')[-1]


@app.route('/')
def index():
    return render_template('principal.html')

@app.route('/manuales')
def manuales():
    return render_template('manuales.html')


@app.route('/uploadajax', methods=['POST'])
def upload_data():
    global data_path
    global columns_names
    global types_data
    file = request.files['file']
    format_file_option = get_extension(file.filename)
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'datos.'+format_file_option))
    data_path = UPLOAD_FOLDER + 'datos.' + format_file_option
    columns_names, counters, types_data, criterion = do.read_data(data_path, format_file_option)
    if criterion is not None:
        return jsonify({'success':  'Data loaded successfully', 'columns_name': columns_names, 'counters': counters, 'types_data': types_data, 'path': data_path, 'criterio': criterion})
    else:
        return jsonify({'error': 'Data error'})


@app.route('/seleccionar', methods=['POST'])
def selection():
    global csv_data_path
    global columns_names
    global types_data
    data = request.form
    data_dictionary = data.copy()
    algorithm_option = data_dictionary['option_algorithm']
    if algorithm_option == 'Chimerge' :
        col_labeled = data_dictionary['labeled_column']
        confidence = float(data_dictionary['confianza'])
        dataframe = do.read_csv_dataframe(csv_data_path)
        numeric_columns = ch.get_numeric_columns(columns_names,types_data)
        discreted_dataframe, process_chimerge = ch.chimerge_general(dataframe, numeric_columns, col_labeled,confidence)
        process, results, suggestion = fs.feature_selection_hamming(discreted_dataframe)
        process = [process_chimerge, process]

    elif algorithm_option == 'Euclidean':
        dataframe = do.read_csv_dataframe(csv_data_path)
        process, results, suggestion = fs.feature_selection_euclidean(dataframe)
    else:
        dataframe = do.read_csv_dataframe(csv_data_path)
        process, results, suggestion = fs.feature_selection_hamming(dataframe)

    if suggestion is not None:
        return jsonify({'success':  'successfully', 'process': process, 'results': results, 'suggestion': suggestion, 'algoritmo': algorithm_option})
    else:
        return jsonify({'error': 'error'})


if __name__ == '__main__':
    app.run(debug=False)
