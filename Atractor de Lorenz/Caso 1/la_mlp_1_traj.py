import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.linear_model
import seaborn as sns
import tensorflow as tf
from mpl_toolkits.mplot3d import Axes3D
import pickle
import os.path as path
from scipy.integrate import odeint
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import InputLayer
from tensorflow import keras
from tensorboard.plugins.hparams import api as hp
from datetime import datetime

# creamos el modelo
def create_model(hparams, capas, units, activ, optim, lrate):

    model = Sequential()
    model.add(InputLayer(1, name='Input_Layer'))
    for i in range(hparams[capas]):
        model.add(Dense(hparams[units], activation=hparams[activ], name=f'dense_{i}'))
    model.add(Dense(3, name='Output_Layer'))

    optimizer_name = hparams[optim]
    learning_rate = hparams[lrate]
    if optimizer_name == "adamax":
        optimizer = tf.keras.optimizers.Adamax(learning_rate=learning_rate)
    elif optimizer_name == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    else:
        raise ValueError("unexpected optimizer name: %r" % (optimizer_name,))
    model.compile(optimizer=hparams[optim],
                loss='mse',
                metrics=["mae"])

    model.summary()
    return model

def train_test_model(model, in_train, out_train, in_test, out_test, x_val, y_val, logdir, hparams, batch_s, epochs, filepath):

    history = model.fit(x=in_train,
                        y=out_train,
                        batch_size=hparams[batch_s],
                        epochs=hparams[epochs],
                        validation_data= (x_val, y_val), 
                        callbacks=[
                        tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1),  # log metrics
                        hp.KerasCallback(logdir, hparams),
                        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=40)],
                        verbose=1)


    # evaluamos las perdidas
    loss, mae= model.evaluate(x=in_test, y=out_test, verbose=2)
    print('Loss:', loss, 'MAE:', mae)

    fig1 = plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validate'], loc='upper right')
    plt.savefig('standard2D_best_Loss.png', dpi=300)


    fig2 = plt.figure(figsize=(10, 6))
    plt.plot(history.history['mae'])
    plt.plot(history.history['val_mae'])
    plt.title('Model mae')
    plt.ylabel('MAE')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validate'], loc='upper right')
    plt.savefig('standard2D_best_MAE.png', dpi=300)

    return loss, mae

def run(run_dir, hparams, capas, units, activ ,optim, lrate, batch_s, epochs, in_train, out_train, in_test, out_test, x_val, y_val, logdir, METRIC_MAE, METRIC_LOSS, filepath):
  with tf.summary.create_file_writer(run_dir).as_default():

    hp.hparams(hparams)  # record the values used in this trial
    model = create_model(hparams, capas, units, activ, optim, lrate)
    loss, mae = train_test_model(model, in_train, out_train, in_test, out_test, x_val, y_val, logdir, hparams, batch_s, epochs, filepath)
    tf.summary.scalar(METRIC_MAE, mae, step=1)
    tf.summary.scalar(METRIC_LOSS, loss, step=1)
    return model


def f(state, t):
    rho = 28.0
    sigma = 10.0
    beta = 8.0 / 3.0
    x, y, z = state  # Desempaqueta el vector de estado
    return sigma * (y - x), x * (rho - z) - y, x * y - beta * z  # Derivadas

def calculate_error(outcome_test, p_test):
    error = []
    for outs in zip(outcome_test, p_test):
        error.append((outs[0] - outs[1])**2)
    return error

def error_relativo(error, outcome_test):
    error_rel = np.sqrt(error) / np.absolute(outcome_test)
    return error_rel

def calculate_error_absolute(outcome_test, p_test):
    error = []
    for outs in zip(outcome_test, p_test):
        error.append(abs(outs[0] - outs[1]))
    return error

def main():

    if path.exists("datos_la_mlp_1_traj.pickle"):
        fichero = open('datos_la_mlp_1_traj.pickle', 'rb')

        # Cargamos los datos del fichero
        X_train = pickle.load(fichero)
        X_test = pickle.load(fichero)
        Y_train = pickle.load(fichero)
        Y_test = pickle.load(fichero)
        X_val = pickle.load(fichero)
        Y_val = pickle.load(fichero)
        scaler = pickle.load(fichero)

    else:

        state0 = [0., 1.0, 1.05]
        t = np.arange(0.0, 20.0, 0.0002)
        tiempo = pd.DataFrame({"Tiempo": t})

        # creamos las variables train y test
        X_train = tiempo.sample(frac=0.8)
        X_train = X_train.sort_values(by='Tiempo')
        X_test = tiempo.drop(X_train.index)

        Y_train = odeint(f, state0, X_train.Tiempo)
        Y_test = odeint(f, state0, X_test.Tiempo)

        Y_test = pd.DataFrame(Y_test, columns={"x": [], "y": [], "z": []})
        Y_train = pd.DataFrame(Y_train, columns={"x": [], "y": [], "z": []})

        print("X_train", X_train)
        print("X_test", X_test)
        print("Y_train", Y_train)
        print("Y_test", Y_test)

        scaler = MinMaxScaler(feature_range=(0, 1))
        # Fit on training set only.
        scaler.fit(Y_train[["x", "y", "z"]])

        # Apply transform to both the training set and the test set.
        Y_train[["x", "y", "z"]] = scaler.transform(Y_train[["x", "y", "z"]])
        Y_test[["x", "y", "z"]] = scaler.transform(Y_test[["x", "y", "z"]])

        X_val = X_train.iloc[0:len(X_train):5]
        Y_val = Y_train.iloc[0:len(X_train):5]

        X_train = X_train.drop(X_val.index)
        Y_train = Y_train.drop(Y_val.index)

        fig = plt.figure(figsize=(7,6))
        fig.tight_layout(pad=0.001)
        ax = fig.gca(projection='3d')
        ax.scatter(Y_train["x"], Y_train["y"], Y_train["z"], s=2)
        ax.scatter(Y_test["x"], Y_test["y"], Y_test["z"], s=2)
        plt.show()

        fig = plt.figure(figsize=(7,6))
        fig.tight_layout(pad=0.001)
        ax = fig.gca(projection='3d')
        ax.scatter(Y_train["x"], Y_train["y"], Y_train["z"], s=2)
        plt.show()


        fig = plt.figure(figsize=(7,6))
        fig.tight_layout(pad=0.001)
        ax = fig.gca(projection='3d')
        ax.scatter(Y_test["x"], Y_test["y"], Y_test["z"], s=2)
        plt.show()


        fig = plt.figure(figsize=(7,6))
        fig.tight_layout(pad=0.2)
        ax = fig.gca(projection='3d')
        ax.plot(Y_test["x"], Y_test["y"], Y_test["z"])
        plt.show()

        fig5 = plt.figure(figsize=(7, 6))
        fig5.tight_layout(pad=0.2)
        plt.plot(Y_test["x"])
        plt.show()

        fig5 = plt.figure(figsize=(7, 6))
        fig5.tight_layout(pad=0.2)
        plt.plot(Y_test["y"])
        plt.show()

        fig5 = plt.figure(figsize=(7, 6))
        fig5.tight_layout(pad=0.2)
        plt.plot(Y_test["z"])
        plt.show()

        pickled_file = open('datos_la_mlp_1_traj.pickle', 'wb')

        pickle.dump(X_train, pickled_file)
        pickle.dump(X_test, pickled_file)
        pickle.dump(Y_train, pickled_file)
        pickle.dump(Y_test, pickled_file)
        pickle.dump(X_val, pickled_file)
        pickle.dump(Y_val, pickled_file)
        pickle.dump(scaler, pickled_file)

        pickled_file.close()

    if path.exists('la_mlp_1_traj.h5'):

        model = keras.models.load_model('la_mlp_1_traj.h5')

    else:

        HP_NUM_CAPAS = hp.HParam("num_capas", hp.Discrete([6]))
        HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([150]))
        HP_ACTIVATION = hp.HParam('activation', hp.Discrete(['relu']))
        HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adamax']))
        HP_L_RATE = hp.HParam('lrate', hp.Discrete([0.01]))
        HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([80]))
        HP_EPOCHS = hp.HParam('epochs', hp.Discrete([600]))

        METRIC_MAE = 'mae'
        METRIC_LOSS = 'loss'

        with tf.summary.create_file_writer('la_mlp_1_traj/logs/hparam_tuning').as_default():
          hp.hparams_config(
            hparams=[HP_NUM_CAPAS, HP_NUM_UNITS,HP_ACTIVATION, HP_OPTIMIZER, HP_L_RATE, HP_BATCH_SIZE, HP_EPOCHS],
            metrics=[hp.Metric(METRIC_MAE, display_name='MAE'),hp.Metric(METRIC_LOSS, display_name='Loss')],
          )

        session_num = 0

        for num_capas in HP_NUM_CAPAS.domain.values:
            for num_units in HP_NUM_UNITS.domain.values:
                for activation in HP_ACTIVATION.domain.values:
                    for optimizer in HP_OPTIMIZER.domain.values:
                        for lrate in HP_L_RATE.domain.values:
                            for batch_size in HP_BATCH_SIZE.domain.values:
                                for epochs in HP_EPOCHS.domain.values:
                                    hparams = {
                                      HP_NUM_CAPAS: num_capas,
                                      HP_NUM_UNITS: num_units,
                                      HP_ACTIVATION: activation,
                                      HP_OPTIMIZER: optimizer,
                                      HP_L_RATE: lrate,
                                      HP_BATCH_SIZE: batch_size,
                                      HP_EPOCHS: epochs
                                    }

                                    log_dir = "la_mlp_1_traj/logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")

                                    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
                                    train_log_dir = 'la_mlp_1_traj/logs/gradient_tape/' + current_time + '/train'
                                    test_log_dir = 'la_mlp_1_traj/logs/gradient_tape/' + current_time + '/test'
                                    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
                                    test_summary_writer = tf.summary.create_file_writer(test_log_dir)

                                    logdir = "la_mlp_1_traj/logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
                                    filepath = 'best_model_biLSTM_1_traj'

                                    print("HP_NUM_UNITS:",hparams)
                                    run_name = "run-%d" % session_num
                                    print('--- Starting trial: %s' % run_name)
                                    print({h.name: hparams[h] for h in hparams})
                                    model = run('la_mlp_1_traj/logs/hparam_tuning/' + run_name,
                                    hparams, HP_NUM_CAPAS, HP_NUM_UNITS, HP_ACTIVATION, HP_OPTIMIZER, HP_L_RATE, HP_BATCH_SIZE, HP_EPOCHS,
                                    X_train, Y_train, X_test, Y_test, X_val, Y_val, logdir, METRIC_MAE, METRIC_LOSS, filepath)

                                    session_num += 1


        model.save('la_mlp_1_traj.h5')

    # predecimos con los punto de test
    p_test = model.predict(X_test)

    #rescalamos los datos predichos y los reales para obtener los valores originales
    p_test = pd.DataFrame(scaler.inverse_transform(p_test), columns=Y_test.columns)
    Y_train = pd.DataFrame(scaler.inverse_transform(Y_train), columns=Y_train.columns)
    Y_test = pd.DataFrame(scaler.inverse_transform(Y_test), columns=Y_test.columns)

    fig = plt.figure(figsize=(7,6))
    plt.subplots_adjust(bottom=0.01, right=0.99, left=0.01, top=0.99)
    ax = fig.gca(projection='3d')
    ax.scatter(Y_train["x"], Y_train["y"], Y_train["z"], s=2)
    ax.set_xlabel('x', fontsize=15)
    ax.set_ylabel('y', fontsize=15)
    ax.set_zlabel('z', fontsize=15)
    ax.tick_params(labelsize=10)
    plt.savefig('train_caso1.png', dpi=300)
    plt.show()


    fig = plt.figure(figsize=(7,6))
    plt.subplots_adjust(bottom=0.01, right=0.99, left=0.01, top=0.99)
    ax = fig.gca(projection='3d')
    ax.scatter(Y_test["x"], Y_test["y"], Y_test["z"], s=2)
    ax.set_xlabel('x', fontsize=15)
    ax.set_ylabel('y', fontsize=15)
    ax.set_zlabel('z', fontsize=15)
    ax.tick_params(labelsize=10)
    plt.savefig('test_caso1.png', dpi=300)
    plt.show()
    

    fig = plt.figure(figsize=(7,6))
    plt.subplots_adjust(bottom=0.01, right=0.99, left=0.01, top=0.99)
    ax = fig.gca(projection='3d')
    ax.scatter(Y_test["x"], Y_test["y"], Y_test["z"], s=1, label="Valor real")
    ax.scatter(p_test["x"], p_test["y"], p_test["z"], s=1, label="Valor predicho")
    ax.set_xlabel('x', fontsize=15)
    ax.set_ylabel('y', fontsize=15)
    ax.set_zlabel('z', fontsize=15)
    ax.tick_params(labelsize=10)
    plt.legend(loc='upper right')
    plt.savefig('resultado_pred_caso1.png', dpi=300)
    plt.show()

    #se calcula el error cuadrático y el absoluto
    error_cuad = np.asarray(calculate_error(Y_test.values, p_test.values))
    error_absoluto = np.asarray(calculate_error_absolute(Y_test.values, p_test.values))


    fig, axs = plt.subplots(2, 3, sharex='col',figsize=(10, 6))
    plt.subplots_adjust(left=0.08, bottom=0.1, right=0.96, top=0.9, wspace=0.35, hspace=None)
    #fig.suptitle('Velocidad de cada coordenada predichas vs reales y error cuadrático')

    axs[0,0].scatter(range(len(Y_test["x"])), Y_test["x"], s=1, label="Valor real")
    axs[0,0].scatter(range(len(Y_test["x"])), p_test["x"], s=1, label="Valor predicho")
    axs[0,0].set_ylabel('x', fontsize=15)
    axs[0,0].legend(loc='upper right')
    axs[0,0].tick_params(axis = 'both', which = 'major', labelsize = 15)

    axs[0,1].scatter(range(len(Y_test["x"])),Y_test["y"], s=1, label="Valor real")
    axs[0,1].scatter(range(len(Y_test["x"])),p_test["y"], s=1, label="Valor predicho")
    axs[0,1].set_ylabel('y', fontsize=15)
    axs[0,1].legend(loc='upper right')
    axs[0,1].tick_params(axis = 'both', which = 'major', labelsize = 15)

    axs[0,2].scatter(range(len(Y_test["x"])),Y_test["z"], s=1, label="Valor real")
    axs[0,2].scatter(range(len(Y_test["x"])),p_test["z"], s=1, label="Valor predicho")
    axs[0,2].set_ylabel('z', fontsize=15)
    axs[0,2].legend(loc='upper right')
    axs[0,2].tick_params(axis = 'both', which = 'major', labelsize = 15)

    axs[1,0].scatter(range(len(error_cuad[:,0])),error_cuad[:,0], s=1, label="Valor real")
    axs[1,0].set_ylabel('Error cuadrático de x', fontsize=15)
    axs[1,0].set_xlabel('Time Steps', fontsize=15)
    axs[1,0].tick_params(axis = 'both', which = 'major', labelsize = 15)

    axs[1,1].scatter(range(len(error_cuad[:,1])),error_cuad[:,2], s=1, label="Valor real")
    axs[1,1].set_ylabel('Error cuadrático de y', fontsize=15)
    axs[1,1].set_xlabel('Time Steps', fontsize=15)
    axs[1,1].tick_params(axis = 'both', which = 'major', labelsize = 15)

    axs[1,2].scatter(range(len(error_cuad[:,2])),error_cuad[:,2], s=1, label="Valor real")
    axs[1,2].set_ylabel('Error cuadrático de z', fontsize=15)
    axs[1,2].set_xlabel('Time Steps', fontsize=15)
    axs[1,2].tick_params(axis = 'both', which = 'major', labelsize = 15)

    plt.savefig('la_mlp_results_error.png', dpi=300)
    plt.show()

    fig, axs = plt.subplots(2, 3, sharex='col',figsize=(10, 6))
    plt.subplots_adjust(left=0.08, bottom=0.1, right=0.96, top=0.9, wspace=0.35, hspace=None)
    #fig.suptitle('Velocidad de cada coordenada predichas vs reales y error cuadrático')

    axs[0,0].scatter(range(len(Y_test["x"])), Y_test["x"], s=1, label="Valor real")
    axs[0,0].scatter(range(len(Y_test["x"])), p_test["x"], s=1, label="Valor predicho")
    axs[0,0].set_ylabel('x', fontsize=15)
    axs[0,0].legend(loc='upper right')
    axs[0,0].tick_params(axis = 'both', which = 'major', labelsize = 15)

    axs[0,1].scatter(range(len(Y_test["x"])),Y_test["y"], s=1, label="Valor real")
    axs[0,1].scatter(range(len(Y_test["x"])),p_test["y"], s=1, label="Valor predicho")
    axs[0,1].set_ylabel('y', fontsize=15)
    axs[0,1].legend(loc='upper right')
    axs[0,1].tick_params(axis = 'both', which = 'major', labelsize = 15)

    axs[0,2].scatter(range(len(Y_test["x"])),Y_test["z"], s=1, label="Valor real")
    axs[0,2].scatter(range(len(Y_test["x"])),p_test["z"], s=1, label="Valor predicho")
    axs[0,2].set_ylabel('z', fontsize=15)
    axs[0,2].legend(loc='upper right')
    axs[0,2].tick_params(axis = 'both', which = 'major', labelsize = 15)

    axs[1,0].scatter(range(len(error_absoluto[:,0])),error_absoluto[:,0], s=1, label="Valor real")
    axs[1,0].set_ylabel('Error absoluto de x', fontsize=15)
    axs[1,0].set_xlabel('Time Steps', fontsize=15)
    axs[1,0].tick_params(axis = 'both', which = 'major', labelsize = 15)

    axs[1,1].scatter(range(len(error_absoluto[:,1])),error_absoluto[:,2], s=1, label="Valor real")
    axs[1,1].set_ylabel('Error absoluto de y', fontsize=15)
    axs[1,1].set_xlabel('Time Steps', fontsize=15)
    axs[1,1].tick_params(axis = 'both', which = 'major', labelsize = 15)

    axs[1,2].scatter(range(len(error_absoluto[:,2])),error_absoluto[:,2], s=1, label="Valor real")
    axs[1,2].set_ylabel('Error absoluto de z', fontsize=15)
    axs[1,2].set_xlabel('Time Steps', fontsize=15)
    axs[1,2].tick_params(axis = 'both', which = 'major', labelsize = 15)

    plt.savefig('la_mlp_results_error_absoluto.png', dpi=300)
    plt.show()


if __name__ == '__main__':
    main()
