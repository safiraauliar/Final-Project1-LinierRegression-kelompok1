from flask import Flask, request, jsonify, render_template
import util

print(util.get_cab_names())
app = Flask(__name__, template_folder='templates')

@app.route('/')
def main():
    return(render_template('app.html'))


@app.route('/get_cab_names', methods=['GET'])
def get_cab_names():
    util.load_saved_artifacts()
    response = jsonify({
        'cab_names': util.get_cab_names()
    })
    response.headers.add('Access-Control-Allow-Origin', '*')

    return response

@app.route('/predict_price', methods=['GET', 'POST'])
def predict_price():
    cab_name = request.form['cab_name']
    distance = float(request.form['distance'])
    surge_multiplier = float(request.form['surge_multiplier'])

    response = jsonify({
        'predicted_price': util.get_predicted_price(cab_name,distance,surge_multiplier)
    })
    response.headers.add('Access-Control-Allow-Origin', '*')

    return response

if __name__ == "__main__":
    print("Starting Python Flask Server For Cab Price Prediction...")
    app.run()