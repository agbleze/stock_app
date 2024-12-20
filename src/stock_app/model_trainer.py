
#%%
from dash import html, Input, Output, State, dcc, callback_context, callback, Dash
from datetime import date
import dash_bootstrap_components as dbc
import pandas as pd
from dash.exceptions import PreventUpdate
#from dashapp.helper_components import (make_boxplot, plot_barplot, get_path)
import dash
import joblib
import functools
import plotly.express as px
import dash_trich_components as dtc
from style import page_style
import yfinance as yf
from pandas.core.indexes.multi import MultiIndex
import functools
from prophet.plot import get_seasonality_plotly_props
from prophet import Prophet
import plotly.graph_objects as go
from prophet.serialize import model_to_json, model_from_json
import json
import numpy as np
import tensorflow as tf
from sklearn import preprocessing
from sklearn import metrics
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Bidirectional, LSTM, Flatten,
                                     TimeDistributed, RepeatVector,
                                     Conv1D, MaxPool1D)
from keras.layers import LSTM, Dense, Bidirectional, MaxPool1D, Dropout
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt


class Transformer(object):
    def __init__(self, data):
        self.data = data
        
    def get_minmax_scaler(self, data=None):
        if data is None:
            data = self.data
        self.minmax_scaler = preprocessing.MinMaxScaler()
        self.minmax_scaler.fit(data)
        return self.minmax_scaler
    
    def transform(self, data, scaler=None):
        if not scaler:
            scaler = self.get_minmax_scaler(data=data)
        transformed_data = scaler.transform(data)
        return transformed_data
  

#%%
class Model_Trainer(object):
    def __init__(self, steps_per_epoch, epochs, 
                 predictors: pd.DataFrame,
                target: pd.DataFrame, start: int,
                train_endpoint: int, window: int, horizon: int,
                validation_steps, monitor='val_loss',
                mode="min", save_model_path='model_store/lstm_multivariate.h5',
                batch_size: int = 64, buffer_size: int = 100,
                model_type="lstm", optimizer="adam", loss="mse"
                ):
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.validation_steps = validation_steps
        self.monitor = monitor
        self.mode = mode
        self.save_model_path = save_model_path
        self.predictors = predictors
        self.target = target
        self.start = start
        self.train_endpoint = train_endpoint
        self.window = window
        self.horizon = horizon
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.model_type = model_type
        self.optimizer = optimizer
        self.loss = loss
        
        tf.random.set_seed(2023)
        np.random.seed(2023)
        
    def horizon_style_data_splitter(self, predictors: pd.DataFrame,
                                    target: pd.DataFrame, start: int,
                                    end: int, window: int, horizon: int
                                    ):
        X = []
        y = []
        start = start + window
        if end is None:
            end = len(predictors) - horizon

        for i in range(start, end):
            indices = range(i-window, i)
            X.append(predictors.iloc[indices])
            indicey = range(i+1, i+1+horizon)
            y.append(target.iloc[indicey])
        return np.array(X), np.array(y)
            
    def timeseries_evaluation_metrics(self, y_true, y_pred):
        mse = metrics.mean_squared_error(y_true, y_pred)
        mae = metrics.mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(metrics.mean_squared_error(y_true, y_pred))
        mape = mean_absolute_percentage_error(y_true, y_pred)
        r2 = metrics.r2_score(y_true, y_pred)
        print('Evaluation metric results:-')
        print(f'MSE is : {mse}')
        print(f'MAE is : {mae}')
        print(f'RMSE is : {rmse}')
        print(f'MAPE is : {mape}')
        print(f'R2 is : {r2}', end='\n\n')
        return {"mean_squared_error": mse, "Mean Absolute Error": mae,
                "root_mean_squared_error": rmse, "Meaan Absolute Percentage Error": mape,
                "R2": r2}

    def plot_loss_history(self, history=None):
        if history is None:
            if hasattr(self, "train_history"):
                history = self.train_history
            else:
                raise ValueError("""history parameter is None and the model has not 
                                    yet been trained. Either provide history parameter or
                                    call the run_model_training to train model before 
                                    plotting loss history
                                """
                                )
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train loss', 'validation loss'], loc='upper left')
        plt.rcParams['figure.figsize'] = [16, 9]
        return plt.show()
    
    def create_model(self, input_shape, optimizer="adam", 
                     loss='mse', model_type="lstm"
                     ):
        if model_type == "lstm":
            self.model = tf.keras.models.Sequential()
            self.model.add(tf.keras.layers.LSTM(units=556, 
                                                    input_shape=input_shape,
                                                    return_sequences=True
                                                    )
                                )
            self.model.add(tf.keras.layers.LSTM(units=556, return_sequences=False))
            self.model.add(tf.keras.layers.Dense(256))
            self.model.add(tf.keras.layers.Dense(256))
            self.model.add(tf.keras.layers.Dense(units=self.horizon))
            self.model.compile(optimizer=optimizer, loss=loss)
        else:
            self.model = Sequential()
            self.model.add(Conv1D(filters=64, kernel_size=3, activation='relu',
                                                input_shape=input_shape
                                                )
                                            )
            self.model.add(MaxPool1D(pool_size=2))
            self.model.add(tf.keras.layers.Dense(556))
            self.model.add(Flatten())
            self.model.add(Dense(556, activation='relu'))
            self.model.add(tf.keras.layers.Dense(256))
            self.model.add(tf.keras.layers.Dense(256))
            self.model.add(Dense(self.horizon))
            self.model.compile(optimizer=optimizer, loss=loss)
        return self.model
    
    def fit_model(self, model, train_data, val_data, save_model_path, epochs, steps_per_epoch,
                  validation_steps=5,
                  monitor='val_loss', mode="min",
                  ):
        self.train_history = model.fit(x=train_data, epochs=epochs,
                                        steps_per_epoch=steps_per_epoch,
                                        validation_data=val_data,
                                        validation_steps=validation_steps, verbose=1,
                                        callbacks=[
                                                    tf.keras.callbacks.ModelCheckpoint(save_model_path,
                                                                                        monitor=monitor,
                                                                                        save_best_only=True,
                                                                                        mode=mode,
                                                                                        verbose=0
                                                                                        )
                                                    ]
                                        )
        return self.train_history
    
    def predict(self, data, model, model_path):
        if model_path:
            model = tf.keras.models.load_model(model_path)
        predicted_results = model.predict(data)
        return predicted_results
    
    def load_data(self, predictor, target, batch_size: int, buffer_size: int, **kwargs):
        loaded_data_slices = tf.data.Dataset.from_tensor_slices((predictor, target))
        self.loaded_data_slices = loaded_data_slices.cache().shuffle(buffer_size).batch(batch_size).repeat()
        return self.loaded_data_slices

    def load_model(self, model_path=None):
        if not model_path:
            model_path = self.save_model_path
        self.loaded_model = tf.keras.models.load_model(model_path)
        return self.loaded_model
        
    
    
    def run_model_training(self, *args, **kwds):
        x_train, y_train = self.horizon_style_data_splitter(predictors=self.predictors, 
                                                            target=self.target,
                                                            start=self.start, end=self.train_endpoint,
                                                            window=self.window,
                                                            horizon=self.horizon
                                                            )

        x_val, y_val = self.horizon_style_data_splitter(predictors=self.predictors, 
                                                        target=self.target,
                                                        start=self.train_endpoint, end=None,
                                                        window=self.window,
                                                        horizon=self.horizon
                                                        )
        train_data = self.load_data(predictor=x_train, target=y_train, batch_size=self.batch_size,
                                    buffer_size=self.buffer_size
                                    )
        val_data = self.load_data(predictor=x_val, target=y_val, batch_size=self.batch_size,
                                  buffer_size=self.buffer_size
                                  )
        if self.model_type == "lstm":
            input_shape = x_train.shape[-2:]
        elif self.model_type == "cnn":
            input_shape = (x_train.shape[1], 
                           x_train.shape[2]
                           )
        model = self.create_model(input_shape=input_shape,
                                optimizer=self.optimizer,
                                loss=self.loss, 
                                model_type=self.model_type
                                )
        
        self.train_history = self.fit_model(model=model, train_data=train_data, val_data=val_data,
                                        epochs=self.epochs, save_model_path=self.save_model_path,
                                        steps_per_epoch=self.steps_per_epoch,
                                        validation_steps=self.validation_steps, monitor=self.monitor,
                                        mode=self.mode
                                        )
        return self.train_history, self.load_model()
        
# %%
data = yf.download("META")

if isinstance(data.columns, MultiIndex):
    data.columns = data.columns.droplevel(1)
# %%
test_df = data.tail(90)
train_df = data.drop(test_df.index)



# %%
trn = Transformer(data=train_df[["Volume"]])
# %%
train_df[["Volume"]] = trn.transform(train_df[["Volume"]])
# %%
test_df[["Volume"]] = trn.minmax_scaler.transform(test_df[["Volume"]])
# %%
predictors = train_df[train_df.columns[1:]]

target = train_df[['Close']]
mod_cls = Model_Trainer(steps_per_epoch=2, epochs=3, 
                        predictors=predictors,
                        target=target, start=0,
                        train_endpoint=int(len(train_df) * 0.75),
                        window=180, horizon=90, validation_steps=5,
                        save_model_path="model_store/META_lstm.h5"
                        )
# %%
train_hist, model = mod_cls.run_model_training()

#%%
mod_cls.plot_loss_history()


# %%
#model.predict(predictors)

data_val = train_df[train_df.columns[1:]].tail(180)
val_rescaled = np.array(data_val).reshape(1, data_val.shape[0], data_val.shape[1])
predicted_results = model.predict(val_rescaled)

#%%

test_df["Close"].to_list()

# %%
mod_cls.timeseries_evaluation_metrics(y_true=test_df["Close"].to_list(),
                                      y_pred=predicted_results.tolist()[0]
                                      )
# %%
