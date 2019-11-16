import pandas as pd
from flask import Flask, jsonify, request
import pickle

# cargar modelo
model = pickle.load(open('svc.pkl','rb'))

# app
app = Flask(__name__)

# routes
@app.route('/', methods=['POST'])
def predict():
    # obtener los datos del body de la peticion
    data = request.get_json(force=True)

    # convertir el json en un dataframe
    data.update((x, [y]) for x, y in data.items())
    data_df = pd.DataFrame.from_dict(data)

    # predecir
    result = model.predict(data_df)

    # estructurar la respuesta
    output = {'results': int(result[0])}

    # retornar los datos
    return jsonify(results=output)

if __name__ == '__main__':
    app.run(port = 5000, debug=True)