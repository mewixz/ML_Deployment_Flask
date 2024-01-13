import flask
import pickle
import pandas as pd
from read import get_plot
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# Use pickle to load in the pre-trained model
with open(f'./model/bike_model_xgboost.pkl', 'rb') as f:
    model = pickle.load(f)

# Initialise the Flask app
app = flask.Flask(__name__, template_folder='templates')


# Root URL 
#@app.get('/') 
#def single_converter(): 
#    # Get the matplotlib plot  
#    plot = get_plot() 
#    plot1 = get_plot() 
  
    # Save the figure in the static directory  
#    plot.savefig(os.path.join('static', 'images', 'plot.png')) 
#    plot1.savefig(os.path.join('static', 'images', 'plot1.png')) 
  
#    return flask.render_template('plot.html') 

    # Get the matplotlib plot  
    #plot1 = get_plot() 
  
    # Save the figure in the static directory  
    #plot1.savefig(os.path.join('static', 'images', 'plot1.png')) 

# Use pickle to load in the pre-trained model
with open(f'model/bike_model_xgboost.pkl', 'rb') as f:
    model = pickle.load(f)

# Initialise the Flask app
app = flask.Flask(__name__, template_folder='templates')

# Set up the main route
@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':

        #    # Get the matplotlib plot  
        #plot = get_plot()   
        # Save the figure in the static directory  
        #plot.savefig(os.path.join('static', 'images', 'plot.png')) 
  
        #return flask.render_template('plot.html') 
        return flask.render_template('index.html') 


        # Just render the initial form, to get input
        # return(flask.render_template('index.html'))
    
    if flask.request.method == 'POST':
        # Extract the input
        temperature = flask.request.form['temperature']
        humidity = flask.request.form['humidity']
        windspeed = flask.request.form['windspeed']

        # Make DataFrame for model
        input_variables = pd.DataFrame([[temperature, humidity, windspeed]],
                                       columns=['temperature', 'humidity', 'windspeed'],
                                       dtype=float,
                                       index=['input'])

        # Get the model's prediction
        prediction = model.predict(input_variables)[0]
    
        # Render the form again, but add in the prediction and remind user
        # of the values they input before
        return flask.render_template('index.html',
                                     original_input={'Temperature':temperature,
                                                     'Humidity':humidity,
                                                     'Windspeed':windspeed},
                                     result=prediction,
                                     )




# Set up the main route

if __name__ == '__main__':
    app.run()