from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM
import numpy as np
import time
from matplotlib import pyplot as plt
from IPython.display import display, clear_output


def sec2time(t):
    h = int(t//3600)
    t = t%3600
    m = int(t//60)
    s = int(t%60)
    return h, m, s


#creating and compiling model
def create_model(train_data, batch_size, neurons, opt, loss_func, out_dim):
    X = train_data[:, :-out_dim]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    model = Sequential()
    model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True, return_sequences=False)) #!!!
    #model.add(LSTM(5, stateful=True))
    model.add(Dense(5, activation = 'tanh'))
    model.add(Dense(out_dim, activation = 'tanh'))
    model.compile(loss=loss_func, optimizer=opt)
    train_loss = [np.nan]
    val_loss = [np.nan]
    epochs = [0]
    train_time = 0
    return model, train_loss, val_loss, epochs, train_time


# fit an network to training data
def fit_rnn(model, train_data, val_data, nb_epoch, batch_size, train_loss, val_loss, epochs, train_time, out_dim):
    X, y = train_data[:, :-out_dim], train_data[:, -out_dim:]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    X_val, y_val = val_data[:, :-out_dim], val_data[:, -out_dim:]
    X_val = X_val.reshape(X_val.shape[0], 1, X_val.shape[1])
    
    fig, ax = plt.subplots()
    last_epoch = epochs[-1]
    t1 = time.time()
    for i in range(nb_epoch):
        H = model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False, validation_data=(X_val, y_val))
        train_loss += H.history['loss']
        val_loss += H.history['val_loss']
        epochs += [last_epoch+i+1]
        model.reset_states()
        ax.cla()
        ax.plot(epochs, train_loss)
        ax.plot(epochs, val_loss)
        ax.set_ylim(bottom = 0)
        ax.set_xlabel('epochs')
        ax.set_ylabel('loss')
        ax.set_title('Model loss by epochs')
        ax.legend(('train_loss','val_loss'))
        ax.grid(True)
        display(fig)
        clear_output(wait = True)
    t2 = time.time() - t1
    train_time += t2

    h,m,s = sec2time(train_time)
    print('The training was completed in {:d}h {:d}m {:d}s ({:.2f} sec/epoch).'.format(h,m,s,train_time/epochs[-1]))
    
    return model, train_loss, val_loss, epochs, train_time

# make a one-step forecast
def forecast_rnn(model, batch_size, X):
    X = X.reshape(1, 1, len(X))
    yhat = model.predict(X, batch_size=batch_size)
    return yhat
