# Recurrent Neural Network


# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import pandas as pd

from rnn_functions import build_series,load_regressor,load_dataset,build_regressor,get_subset,apply_inverse_transform,plot_stock_prices


stock_file='../Datasets/RNN/Google_Stock_Price'
n_obs=60
n_test_examples=50
regressor_filename=None
features=['Open','High','Low']
rnn_structure=(50,50,50,50)
n_epochs=100


training_set,training_set_scaled,sc,real_stock_price,real_stock_price_scaled=load_dataset('{}.PA.csv'.format(stock_file),n_test_examples,features)

# Creating a data structure with 60 timesteps and 1 output
X_train = build_series(training_set_scaled,n_obs)

id_price=0
y_train = np.array(training_set_scaled[n_obs:,id_price])


# Part 2 - Build or load the RNN

if regressor_filename is None:
    # Part 2 - Building the RNN    
    regressor=build_regressor(rnn_structure,(X_train.shape[1], X_train.shape[2]),dropout=0.2)
    
    # Fitting the RNN to the Training set
    regressor.fit(X_train, y_train, epochs = n_epochs, batch_size = 32)
else:
    regressor=load_regressor(regressor_filename)


# Part 3 - Making the predictions and visualising the results

# Getting the predicted stock price
inputs=get_subset(training_set_scaled,real_stock_price_scaled,n_obs)

X_test=build_series(inputs,n_obs)

predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = apply_inverse_transform( predicted_stock_price, X_train.shape[2], sc )

# Visualising the results
plot_stock_prices(real_stock_price,predicted_stock_price)
