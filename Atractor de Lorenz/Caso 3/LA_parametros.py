import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import os
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D
import pickle
import os.path as path
from scipy.integrate import odeint
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Input
from tensorflow import keras
from tensorboard.plugins.hparams import api as hp
from datetime import datetime
import seaborn as sns


# En este codigo solo entrenamos el modelo con una serie de trayectorias, es decir,
# para un determinado intervalo de tiempo y unos parámetros variables a,b,c


# creamos el modelo
def create_model(hparams, capas, units, activ, optim, lrate):

    model = Sequential()
    model.add(Input(shape=(2500,), name='Input_1'))
    model.add(Input(shape=(2500,), name='Input_2'))
    model.add(Input(shape=(2500,), name='Input_3'))
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

def train_test_model(model, in_train, out_train, in_test, out_test, logdir, hparams, batch_s, epochs, filepath):

    history = model.fit(x=in_train,
                        y=out_train,
                        batch_size=hparams[batch_s],
                        epochs=hparams[epochs],
                        validation_split= 0.2, 
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

def run(run_dir, hparams, capas, units, activ ,optim, lrate, batch_s, epochs, in_train, out_train, in_test, out_test, logdir, METRIC_MAE, METRIC_LOSS, filepath):
  with tf.summary.create_file_writer(run_dir).as_default():

    hp.hparams(hparams)  # record the values used in this trial
    model = create_model(hparams, capas, units, activ, optim, lrate)
    loss, mae = train_test_model(model, in_train, out_train, in_test, out_test, logdir, hparams, batch_s, epochs, filepath)
    tf.summary.scalar(METRIC_MAE, mae, step=1)
    tf.summary.scalar(METRIC_LOSS, loss, step=1)
    return model


def f(state, t, a, b, c):
    rho = a
    sigma = b
    beta = c
    x, y, z = state  # Desempaqueta el vector de estado
    return sigma * (y - x), x * (rho - z) - y, x * y - beta * z  # Derivadas


def calculate_error(outcome_test, p_test):
    error = []
    for outs in zip(outcome_test, p_test):
        error.append((outs[0] - outs[1])**2)
    return error

def calculate_error_absolute(outcome_test, p_test):
    error = []
    for outs in zip(outcome_test, p_test):
        error.append(abs(outs[0] - outs[1]))
    return error


def error_relativo(error, outcome_test):
    error_rel = np.sqrt(error) / np.absolute(outcome_test)
    return error_rel

def feature_analysis_numerical(df, var_name, n_bins=30, quantiles=False,
                               minV=float('-inf'), maxV=float('inf'), path_fig=None, dpi=125):


    df_local = df.copy()
    df_local = df_local[[var_name]].dropna()
    df_local = df_local[(df_local[var_name] >= minV) & (df_local[var_name] <= maxV)]

    if quantiles:
        bins = pd.qcut(df_local[var_name], q=n_bins, duplicates='drop')
    else:
        bins = pd.cut(df_local[var_name], bins=n_bins, include_lowest=True)
   
    df_local['bin'] = bins

    grouped_df = df_local.groupby('bin').agg({'bin': ['count']})
    grouped_df.reset_index(inplace=True)
    grouped_df.columns = ['bin', 'count']

    left_interval = lambda x: x.left
    grouped_df['inf'] = list(map(left_interval, grouped_df['bin']))

    grouped_df.sort_values(by=['inf'], inplace=True)

    plt.style.use('seaborn')
    _, ax = plt.subplots(figsize=(18, 6))

    g = sns.barplot(x=grouped_df['bin'], y=grouped_df["count"], ax=ax, alpha=0.6)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
    g.set_yscale("log")
    ticks = [1, 10, 100, 1000]
    g.set_yticks(ticks)
    g.set_yticklabels(ticks)
    ax.tick_params(axis = 'x', which = 'major', labelsize = 40)
    ax.tick_params(axis = 'y', which = 'major', labelsize = 40)
    ax.set_xlabel("Error absoluto", fontsize=50)
    ax.set_ylabel("", fontsize=30)
    #ax.set_title(f"Distribution of variable {var_name}")

    if path_fig:
        plt.savefig(path_fig + '.png', bbox_inches="tight", dpi=dpi)
        plt.close('All')

    plt.show()


def create_dataset(X_t, state0):

    Y_tr = pd.DataFrame({'rho': [], 'sigma': [], 'beta': []})
    Y_tra = []

    x_t = []
    y_t = []
    z_t = []

    for row in X_t[0].itertuples(index=True, name='Pandas'):
        Y_tra = (odeint(f, state0, X_t[1].Tiempo, args=(
            getattr(row, 'rho'), getattr(row, 'sigma'), getattr(row, 'beta'))))
        x_val = [x[0] for x in Y_tra]
        y_val = [x[1] for x in Y_tra]
        z_val = [x[2] for x in Y_tra]
        x_t.append(x_val)
        y_t.append(y_val)
        z_t.append(z_val)

        df = pd.DataFrame({'rho': [getattr(row, 'rho')], 'sigma': [
                          getattr(row, 'sigma')], 'beta': [getattr(row, 'beta')]})
        Y_tr = pd.concat([Y_tr, df])

    x_t = np.asarray(x_t)
    y_t = np.asarray(y_t)
    z_t = np.asarray(z_t)

    inputs_train = [x_t, y_t, z_t]

    return inputs_train, Y_tr



def main():
    if path.exists("datos_la_param_2.pickle"):
        fichero = open('datos_la_param_2.pickle', 'rb')

        # Cargamos los datos del fichero
        x_train = pickle.load(fichero)
        x_test = pickle.load(fichero)
        y_train = pickle.load(fichero)
        y_test = pickle.load(fichero)
        scaler = pickle.load(fichero)

    else:

        a = np.random.uniform(0, 30, 30)
        b = np.random.uniform(0, 30, 30)
        c = np.random.uniform(0, 30, 30)

        param = pd.DataFrame(np.array(np.meshgrid(
            a, b, c)).T.reshape(-1, 3), columns=['rho', 'sigma', 'beta'])
        print(param)

        state0 = [0., 1.0, 1.05]

        t = np.arange(0.0, 5.0, 0.002)

        tiempo = pd.DataFrame({"Tiempo": t})
        print(tiempo.shape)

        # creamos las variables train y test
        param_train = param.sample(frac=0.8)
        param_test = param.drop(param_train.index)
        X_train = (param_train, tiempo)
        X_test = [param_test, tiempo]

        x_train, y_train = create_dataset(X_train, state0)
        x_test, y_test = create_dataset(X_test, state0)

        print(x_train)

        scaler = MinMaxScaler(feature_range=(0, 1))
        # Fit on training set only.
        scaler.fit(y_train[['rho', 'sigma', 'beta']])

        # Apply transform to both the training set and the test set.
        y_train[['rho', 'sigma', 'beta']] = scaler.transform(y_train[['rho', 'sigma', 'beta']])
        y_test[['rho', 'sigma', 'beta']] = scaler.transform(y_test[['rho', 'sigma', 'beta']])

        pickled_file = open('datos_la_param_2.pickle', 'wb')

        pickle.dump(x_train, pickled_file)
        pickle.dump(x_test, pickled_file)
        pickle.dump(y_train, pickled_file)
        pickle.dump(y_test, pickled_file)
        pickle.dump(scaler, pickled_file)

        pickled_file.close()

    if path.exists('la_parma_2.h5'):

        model = keras.models.load_model('la_parma_2.h5')

    else:

        HP_NUM_CAPAS = hp.HParam("num_capas", hp.Discrete([6]))
        HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([300]))
        HP_ACTIVATION = hp.HParam('activation', hp.Discrete(['relu']))
        HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adamax']))
        HP_L_RATE = hp.HParam('lrate', hp.Discrete([0.0001]))
        HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([20]))
        HP_EPOCHS = hp.HParam('epochs', hp.Discrete([300]))

        METRIC_MAE = 'mae'
        METRIC_LOSS = 'loss'

        with tf.summary.create_file_writer('la_parma/logs/hparam_tuning').as_default():
          hp.hparams_config(
            hparams=[HP_NUM_CAPAS, HP_NUM_UNITS,HP_ACTIVATION, HP_OPTIMIZER, HP_L_RATE, HP_BATCH_SIZE, HP_EPOCHS],
            metrics=[hp.Metric(METRIC_MAE, display_name='MAE'),hp.Metric(METRIC_LOSS, display_name='Loss')],
          )

        session_num = 14

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

                                    log_dir = "la_parma_2/logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")

                                    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
                                    train_log_dir = 'la_parma_2/logs/gradient_tape/' + current_time + '/train'
                                    test_log_dir = 'la_parma_2/logs/gradient_tape/' + current_time + '/test'
                                    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
                                    test_summary_writer = tf.summary.create_file_writer(test_log_dir)

                                    logdir = "la_parma_2/logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
                                    filepath = 'best_model_biLSTM_1_traj'

                                    print("HP_NUM_UNITS:",hparams)
                                    run_name = "run-%d" % session_num
                                    print('--- Starting trial: %s' % run_name)
                                    print({h.name: hparams[h] for h in hparams})
                                    model = run('la_parma_2/logs/hparam_tuning/' + run_name,
                                    hparams, HP_NUM_CAPAS, HP_NUM_UNITS, HP_ACTIVATION, HP_OPTIMIZER, HP_L_RATE, HP_BATCH_SIZE, HP_EPOCHS,
                                    x_train, y_train, x_test, y_test, logdir, METRIC_MAE, METRIC_LOSS, filepath)

                                    session_num += 1


        model.save('la_parma_2.h5')

    p_test = model.predict(x_test)

    y_test = pd.DataFrame(scaler.inverse_transform(y_test), columns=y_test.columns)
    p_test = pd.DataFrame(scaler.inverse_transform(p_test), columns=y_test.columns)

    y_train = pd.DataFrame(scaler.inverse_transform(y_train), columns=y_train.columns)

    error_absoluto = np.asarray(calculate_error_absolute(y_test.values, p_test.values))
    error_cuad = np.asarray(calculate_error(y_test.values, p_test.values))

    for i in range(0,30,1):

        print("Parametros reales: " , "a: " , round(y_test.loc[i, "rho"], 6) , "b: " , round(y_test.loc[i, "sigma"],6) , "c: " , round(y_test.loc[i, "beta"],6))
        print("Parametros predic: " , "a: " , + p_test.loc[i, "rho"] , "b: " , p_test.loc[i, "sigma"] , "c: " , p_test.loc[i, "beta"])
        print("Error absoluto: " , "a: " , + error_absoluto[i, 0] , "b: " , error_absoluto[i, 1] , "c: " , error_absoluto[i, 2])
        print("\n")

    error = calculate_error(y_test.values, p_test.values)
    error_absoluto = pd.DataFrame(error_absoluto, columns=["a","b","c"])

    fig7 = plt.figure(figsize=(9, 6))
    plt.scatter(range(len(error_absoluto["a"])),error_absoluto["a"], s=1)
    plt.xlabel('a', fontsize=20)
    plt.ylabel('Error absoluto', fontsize=20)
    plt.yticks(fontsize=15)
    plt.xticks(fontsize=15)
    plt.savefig('param_a.png', dpi=300)
    plt.show()

    feature_analysis_numerical(error_absoluto, "a", n_bins=15, quantiles=False,
                               minV=0.0, maxV=float('inf'), path_fig="a_error_abs", dpi=125)


    feature_analysis_numerical(error_absoluto, "b", n_bins=15, quantiles=False,
                               minV=0.0, maxV=float('inf'), path_fig="b_error_abs", dpi=125)


    feature_analysis_numerical(error_absoluto, "c", n_bins=15, quantiles=False,
                               minV=0.0, maxV=float('inf'), path_fig="c_error_abs", dpi=125)



if __name__ == '__main__':
    main()
