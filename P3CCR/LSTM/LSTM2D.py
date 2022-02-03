import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
import os.path as path
import pickle
from keras.models import Model, Sequential
from keras.layers import LSTM, Dense, Bidirectional, Dropout
from tensorboard.plugins.hparams import api as hp
from datetime import datetime

def scaler_data(columns, data_train, data_test):

    scaler = MinMaxScaler(feature_range=(0, 1))

    # Ajustamos solo a los datos de entrenamiento.
    scaler.fit(data_train[columns])

    # Aplicamos la transformación tantos al conjunto de entrenamiento como al de test.
    data_train[columns] = scaler.transform(
        data_train[columns])
    data_test[columns] = scaler.transform(
        data_test[columns])

    return scaler, data_train, data_test

#esta función crea el modelo con los hiperparámetros fijados
def create_model(hparams, capas, units, dropout, window, optim, lrate):

    model = Sequential()
    if hparams[capas] > 0:
        for i in range(hparams[capas]-1):
            model.add(LSTM(hparams[units], return_sequences=True, input_shape=(hparams[window], 4)))
            model.add(Dropout(hparams[dropout]))
    model.add(LSTM(hparams[units], input_shape=(hparams[window], 4)))
    model.add(Dropout(hparams[dropout]))
    model.add(Dense(4, name='Output_Layer'))

    optimizer_name = hparams[optim]
    learning_rate = hparams[lrate]
    if optimizer_name == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name == "adamax":
        optimizer = tf.keras.optimizers.Adamax(learning_rate=learning_rate)
    else:
        raise ValueError("unexpected optimizer name: %r" % (optimizer_name,))
    model.compile(optimizer=hparams[optim],
                loss='mse',
                metrics=["mae"])

    model.summary()
    return model

#esta función entrena el modelo y evalua las pérdidas
def train_test_model(model, in_train, out_train, in_test, out_test, logdir, hparams, batch_s, epochs, filepath):

    history = model.fit(x=in_train,
                        y=out_train,
                        batch_size= hparams[batch_s],
                        epochs= hparams[epochs],
                        validation_split = 0.2, 
                        callbacks=[
                        tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1, write_graph=True),  # log metrics
                        hp.KerasCallback(logdir, hparams),  # log hparams
                        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10),],
                        verbose=1)


    # evaluamos las perdidas
    loss, mae= model.evaluate(x=in_test, y=out_test, verbose=2)
    print('Loss:', loss, 'MAE:', mae)

    #Representamos los valores de pérdida de entrenamiento y validación
    fig1 = plt.figure(figsize=(12, 9))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validate'], loc='upper right')
    plt.savefig('CR3BP1trajLSTM_best_Loss.png', dpi=300)

    #Representamos los valores del mae de entrenamiento y validación
    fig2 = plt.figure(figsize=(12, 9))
    plt.plot(history.history['mae'])
    plt.plot(history.history['val_mae'])
    plt.title('Model MAE')
    plt.ylabel('MAE')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validate'], loc='upper right')
    plt.savefig('CR3BP1trajLSTM_best_MAE.png', dpi=300)

    return loss, mae

# creamos la funcion para calcular el error cuadratico
def calculate_error(outcome_test, p_test):
    error = []
    for outs in zip(outcome_test, p_test):
        error.append((outs[0] - outs[1])**2)
    return error

# creamos la funcion para calcular el error absoluto
def calculate_error_absolute(outcome_test, p_test):
    error = []
    for outs in zip(outcome_test, p_test):
        error.append(abs(outs[0] - outs[1]))
    return error

# creamos la funcion para crear las ventanas y la salida
def create_dataset(dataset, window):
    inputs, output = [], []
    for i in range(window, len(dataset)):
        inputs.append(dataset[i-window:i])
        output.append(dataset[i])
    return np.array(inputs), np.array(output)

#esta funcion llama a las funciones que crean el modelo y lo entrenan
def run(run_dir, hparams, capas, units, dropout, window, optim,
    lrate, batch_s, epochs, in_train, out_train,in_test, out_test, logdir,METRIC_MAE,METRIC_LOSS, filepath):

    with tf.summary.create_file_writer(run_dir).as_default():

        hp.hparams(hparams)  # record the values used in this trial
        model = create_model(hparams, capas, units, dropout, window ,optim, lrate)
        loss, mae = train_test_model(model, in_train, out_train, in_test, out_test,
                                     logdir, hparams, batch_s, epochs, filepath)
        tf.summary.scalar(METRIC_MAE, mae, step=1)
        tf.summary.scalar(METRIC_LOSS, loss, step=1)
        return model

def main():

    #si tenemos un modelo entrenado
    if path.exists('model_LSTM2D_best.h5'):

        fichero = open('datosLSTM2D.pickle', 'rb')

        # Cargamos los datos del fichero
        data_train = pickle.load(fichero)
        data_test = pickle.load(fichero)
        input_train = pickle.load(fichero)
        output_train = pickle.load(fichero)
        input_test = pickle.load(fichero)
        output_test = pickle.load(fichero)
        scaler = pickle.load(fichero)
        window = pickle.load(fichero)

        #cargamos el modelo ya entrenado
        model = keras.models.load_model('model_LSTM2D_best.h5')

    #si no tenemos ningun modelo
    else:

        filename_traj = '../fullTrajectories/1/trajectories.csv'

        #leemos el archivo con las trayectorias
        data_traj = pd.read_csv(filename_traj, names=[
                                 "TimeStep", "x", "y", "vx", "vy"]) 

        data = pd.DataFrame({"TimeStep": [], "x": [], "y": [], "vx": [], "vy": []})
        data = data_traj.iloc[0:100000:2] #seleccionamos 1 de cada 2 puntos de la trayectoria
        
        data_train = data.iloc[0:50000:2] #seleccionamos 1 de cada 2 puntos de los datos restantes para entrenamiento
        data_test = data.drop(data_train.index) #los datos restantes se usan de test

        #solo se utilizan para entrenar y testar los valores de las coordenadas y velocidades
        data_train = data_train[["x", "y", "vx", "vy"]]
        data_test = data_test[["x", "y", "vx", "vy"]]

        data_train = data_train.reset_index(drop=True)
        data_test = data_test.reset_index(drop=True)


        #representamos las coordenadas de entrenamiento y de test
        fig = plt.figure()
        plt.scatter(data_train.iloc[0:25000:2]["x"], data_train.iloc[0:25000:2]["y"], label="Entrenamiento", s=2)
        plt.scatter(data_test.iloc[1:25000:2]["x"], data_test.iloc[1:25000:2]["y"], label="Test", s=2)
        #plt.title("Coordenadas reales vs predichas en 2D")
        plt.xlabel("x",fontsize=15)
        plt.ylabel("y",fontsize=15)
        plt.yticks(fontsize=12)
        plt.xticks(fontsize=12)
        plt.legend(loc='upper right',fontsize=12)
        plt.savefig('LSTM2D_coord_test_train.pdf', dpi=300) 
        plt.show()

        #representamos las velocidades de entrenamiento y de test
        fig = plt.figure()
        plt.scatter(data_train.iloc[0:25000:2]["vx"], data_train.iloc[0:25000:2]["vy"], label="Entrenamiento", s=2)
        plt.scatter(data_test.iloc[1:25000:2]["vx"], data_test.iloc[1:25000:2]["vy"], label="Test", s=2)
        plt.xlabel("vx",fontsize=15)
        plt.ylabel("vy",fontsize=15)
        plt.yticks(fontsize=12)
        plt.xticks(fontsize=12)
        plt.legend(loc='upper right',fontsize=12)
        plt.savefig('LSTM2D_vel_test_train.pdf', dpi=300) 
        plt.show()

        #escalamos los datos
        scaler, data_train, data_test = scaler_data(["x", "y", "vx", "vy"], data_train, data_test)

        data_train = np.asarray(data_train)
        data_test = np.asarray(data_test)


        #se seleccionan aquellos hiperparámetros y las métricas con los que se quieran ensayar
        HP_NUM_CAPAS = hp.HParam('num_capas', hp.Discrete([1]))
        HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([200]))
        HP_DROPOUT = hp.HParam('dropout', hp.Discrete([0.2]))
        HP_WINDOW = hp.HParam('window', hp.Discrete([4]))
        HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adamax']))
        HP_L_RATE = hp.HParam('lrate', hp.Discrete([0.0001]))
        HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([60]))
        HP_EPOCHS = hp.HParam('epochs', hp.Discrete([300]))

        METRIC_MAE = 'mae'
        METRIC_LOSS = 'loss'

        with tf.summary.create_file_writer('LSTM2D/logs/hparam_tuning').as_default():
          hp.hparams_config(
            hparams=[HP_NUM_CAPAS, HP_NUM_UNITS, HP_DROPOUT, HP_WINDOW, HP_OPTIMIZER, HP_L_RATE, HP_BATCH_SIZE, HP_EPOCHS],
            metrics=[hp.Metric(METRIC_MAE, display_name='MAE'),hp.Metric(METRIC_LOSS, display_name='Loss')],
          )

        # mediante los bucles anidades se llevan a cabo todos los ensayos producto de las combinaciones con los
        # hiperparámetros elegidos
        session_num = 50
        for num_capas in HP_NUM_CAPAS.domain.values:
            for num_units in HP_NUM_UNITS.domain.values:
                for dropout in HP_DROPOUT.domain.values:
                    for window in HP_WINDOW.domain.values:
                        for optimizer in HP_OPTIMIZER.domain.values:
                            for lrate in HP_L_RATE.domain.values:
                                for batch_size in HP_BATCH_SIZE.domain.values:
                                    for epochs in HP_EPOCHS.domain.values:
                                        hparams = {
                                          HP_NUM_CAPAS: num_capas,
                                          HP_NUM_UNITS: num_units,
                                          HP_DROPOUT: dropout,
                                          HP_WINDOW: window,
                                          HP_OPTIMIZER: optimizer,
                                          HP_L_RATE: lrate,
                                          HP_BATCH_SIZE: batch_size,
                                          HP_EPOCHS: epochs
                                        }


                                        input_train, output_train = create_dataset(data_train, window)
                                        input_train = input_train.reshape(
                                        input_train.shape[0], input_train.shape[1], 4)

                                        print("input_train",input_train)
                                        print("output_train",output_train)

                                        input_test, output_test = create_dataset(data_test, window)
                                        input_test = input_test.reshape(
                                        input_test.shape[0], input_test.shape[1], 4)

                                        print("input_test",input_test)
                                        print("output_test",output_test)

                                        log_dir = "LSTM2D/logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")

                                        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
                                        train_log_dir = 'LSTM2D/logs/gradient_tape/' + current_time + '/train'
                                        test_log_dir = 'LSTM2D/logs/gradient_tape/' + current_time + '/test'
                                        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
                                        test_summary_writer = tf.summary.create_file_writer(test_log_dir)

                                        logdir = "LSTM2D/logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
                                        filepath = 'best_model_biLSTM_1_traj'

                                        print("HP_NUM_UNITS:",hparams)
                                        run_name = "run-%d" % session_num
                                        print('--- Starting trial: %s' % run_name)
                                        print({h.name: hparams[h] for h in hparams})
                                        model = run('LSTM2D/logs/hparam_tuning/' + run_name,
                                        hparams, HP_NUM_CAPAS, HP_NUM_UNITS, HP_DROPOUT, HP_WINDOW, HP_OPTIMIZER, HP_L_RATE, HP_BATCH_SIZE, HP_EPOCHS,
                                        input_train, output_train, input_test, output_test, logdir,METRIC_MAE,METRIC_LOSS, filepath)


                                        session_num += 1

        #guardamos las variables que nos interesan
        pickled_file = open('datosLSTM2D.pickle', 'wb')

        pickle.dump(data_train, pickled_file)
        pickle.dump(data_test, pickled_file)
        pickle.dump(input_train, pickled_file)
        pickle.dump(output_train, pickled_file)
        pickle.dump(input_test, pickled_file)
        pickle.dump(output_test, pickled_file)
        pickle.dump(scaler, pickled_file)
        pickle.dump(window, pickled_file)

        pickled_file.close()

        #guardamos el modelo
        model.save('model_LSTM2D_best.h5')

    # predecimos con los punto de test
    p_test = model.predict(input_test[:5000])

    #rescalamos los datos predichos y los reales para obtener los valores originales
    p_test = pd.DataFrame(scaler.inverse_transform(p_test), columns=["x", "y","vx","vy"])
    output_test = pd.DataFrame(scaler.inverse_transform(output_test[:5000]), columns=["x", "y","vx","vy"])

    #se calcula el error cuadrático y el absoluto
    error_cuad = np.asarray(calculate_error(output_test.values, p_test.values))
    error_absoluto = np.asarray(calculate_error_absolute(output_test.values, p_test.values))

    #representamos las coordenadas predichas vs los valores reales
    fig = plt.figure()
    plt.scatter(output_test["x"] , output_test["y"], label="Valor real", s=2)
    plt.scatter(p_test["x"], p_test["y"], label="Predicción", s=2)
    plt.xlabel("x",fontsize=15)
    plt.ylabel("y",fontsize=15)
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.legend(loc='upper right',fontsize=12)
    plt.savefig('LSTM2D_coord.pdf', dpi=300) 
    plt.show()

    #representamos las velocidades predichas vs los valores reales
    fig = plt.figure()
    plt.scatter(output_test["vx"] , output_test["vy"],label="Valor real", s=2)
    plt.scatter(p_test["vx"], p_test["vy"],label="Predicción", s=2)
    plt.xlabel("vx",fontsize=15)
    plt.ylabel("vy",fontsize=15)
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.legend(loc='upper right',fontsize=12)
    plt.savefig('LSTM2D_vel.pdf', dpi=300) 
    plt.show()


    
    fig, axs = plt.subplots(2, 2, sharex='col',figsize=(10, 6))
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.96, top=0.9, wspace=0.3, hspace=None)

    axs[0,0].scatter(range(len(output_test["x"])),output_test["x"], label="Valor real de x", s=2)
    axs[0,0].scatter(range(len(output_test["x"])),p_test["x"], label="Predicción de x", s=2)
    axs[0,0].set_ylabel('x',fontsize=15)
    axs[0,0].tick_params(labelsize=10)
    axs[0,0].legend(loc='upper right')

    axs[0,1].scatter(range(len(output_test["x"])),output_test["y"], label="Valor real de y", s=2)
    axs[0,1].scatter(range(len(output_test["x"])),p_test["y"], label="Predicción de y", s=2)
    axs[0,1].set_ylabel('y',fontsize=15)
    axs[0,1].tick_params(labelsize=10)
    axs[0,1].legend(loc='upper right')

    axs[1,0].scatter(range(len(output_test["x"])),error_cuad[:, 0], s=2)
    axs[1,0].set_xlabel('TimeStep',fontsize=15)
    axs[1,0].set_ylabel('x error cuadrático',fontsize=15)
    axs[1,0].tick_params(labelsize=10)

    axs[1,1].scatter(range(len(output_test["x"])),error_cuad[:, 1], s=2)
    axs[1,1].set_xlabel('TimeStep',fontsize=15)
    axs[1,1].set_ylabel('y error cuadrático',fontsize=15)
    axs[1,1].tick_params(labelsize=10)

    plt.savefig('LSTM2D_best_Pred_coord.png', dpi=300)
    plt.show()

    
    fig, axs = plt.subplots(2, 2, sharex='col',figsize=(10, 6))
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.96, top=0.9, wspace=0.3, hspace=None)

    axs[0,0].scatter(range(len(output_test["vx"])), output_test["vx"], label="Valor real de vx", s=2)
    axs[0,0].scatter(range(len(output_test["vx"])),p_test["vx"], label="Predicción de vx", s=2)
    axs[0,0].set_ylabel('vx',fontsize=15)
    axs[0,0].tick_params(labelsize=10)
    axs[0,0].legend(loc='upper right')

    axs[0,1].scatter(range(len(output_test["vx"])),output_test["vy"], label="Valor real de vy", s=2)
    axs[0,1].scatter(range(len(output_test["vx"])),p_test["vy"], label="Predicción de vy", s=2)
    axs[0,1].set_ylabel('vy',fontsize=15)
    axs[0,1].tick_params(labelsize=10)
    axs[0,1].legend(loc='upper right')

    axs[1,0].scatter(range(len(output_test["vx"])),error_cuad[:, 2], s=2)
    axs[1,0].set_ylabel('vx error cuadrático',fontsize=15)
    axs[1,0].set_xlabel('TimeStep',fontsize=15)
    axs[1,0].tick_params(labelsize=10)

    axs[1,1].scatter(range(len(output_test["vx"])),error_cuad[:, 3], s=2)
    axs[1,1].set_ylabel('vy error cuadrático',fontsize=15)
    axs[1,1].set_xlabel('TimeStep',fontsize=15)
    axs[1,1].tick_params(labelsize=10)

    plt.savefig('LSTM2D_best_Pred_vel.png', dpi=300)
    plt.show()

    #desglosamos cada coordenada y la representamos con su error absoluto
    fig, axs = plt.subplots(2, 2, sharex='col',figsize=(10, 6))
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.96, top=0.9, wspace=0.3, hspace=None)
    #fig.suptitle('Coordenadas predichas vs reales y error cuadrático',fontsize=20)

    axs[0,0].scatter(range(len(output_test["x"])),output_test["x"], label="Valor real de x", s=2)
    axs[0,0].scatter(range(len(output_test["x"])),p_test["x"], label="Predicción de x", s=2)
    axs[0,0].set_ylabel('x',fontsize=15)
    axs[0,0].tick_params(labelsize=10)
    axs[0,0].legend(loc='upper right')

    axs[0,1].scatter(range(len(output_test["x"])),output_test["y"], label="Valor real de y", s=2)
    axs[0,1].scatter(range(len(output_test["x"])),p_test["y"], label="Predicción de y", s=2)
    axs[0,1].set_ylabel('y',fontsize=15)
    axs[0,1].tick_params(labelsize=10)
    axs[0,1].legend(loc='upper right')

    axs[1,0].scatter(range(len(output_test["x"])),error_absoluto[:, 0], s=2)
    axs[1,0].set_xlabel('TimeStep',fontsize=15)
    axs[1,0].set_ylabel('x error absoluto',fontsize=15)
    axs[1,0].tick_params(labelsize=10)

    axs[1,1].scatter(range(len(output_test["x"])),error_absoluto[:, 1], s=2)
    axs[1,1].set_xlabel('TimeStep',fontsize=15)
    axs[1,1].set_ylabel('y error absoluto',fontsize=15)
    axs[1,1].tick_params(labelsize=10)

    plt.savefig('LSTM2D_best_Pred_coord_error_abs.png', dpi=300)
    plt.show()

    #desglosamos cada componente de la velocidad y la representamos con su error absoluto
    fig, axs = plt.subplots(2, 2, sharex='col',figsize=(10, 6))
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.96, top=0.9, wspace=0.3, hspace=None)

    axs[0,0].scatter(range(len(output_test["vx"])), output_test["vx"], label="Valor real de vx", s=2)
    axs[0,0].scatter(range(len(output_test["vx"])),p_test["vx"], label="Predicción de vx", s=2)
    axs[0,0].set_ylabel('vx',fontsize=15)
    axs[0,0].tick_params(labelsize=10)
    axs[0,0].legend(loc='upper right')

    axs[0,1].scatter(range(len(output_test["vx"])),output_test["vy"], label="Valor real de vy", s=2)
    axs[0,1].scatter(range(len(output_test["vx"])),p_test["vy"], label="Predicción de vy", s=2)
    axs[0,1].set_ylabel('vy',fontsize=15)
    axs[0,1].tick_params(labelsize=10)
    axs[0,1].legend(loc='upper right')

    axs[1,0].scatter(range(len(output_test["vx"])),error_absoluto[:, 2], s=2)
    axs[1,0].set_ylabel('vx error absoluto',fontsize=15)
    axs[1,0].set_xlabel('TimeStep',fontsize=15)
    axs[1,0].tick_params(labelsize=10)

    axs[1,1].scatter(range(len(output_test["vx"])),error_absoluto[:, 3], s=2)
    axs[1,1].set_ylabel('vy error absoluto',fontsize=15)
    axs[1,1].set_xlabel('TimeStep',fontsize=15)
    axs[1,1].tick_params(labelsize=10)

    plt.savefig('LSTM2D_best_Pred_vel_error_abs.png', dpi=300)
    plt.show()


    f = h5py.File('model_LSTM2D_best.h5', 'r')
    dset = f['key']
    data = np.array(dset[:,:,:])
    file = 'test.jpg'
    cv2.imwrite(file, data)

"""
    window = window
    inputs, output2, output = [], [], []
    inputs = data_test[0:window]    
    inputs = inputs.reshape(1, window, 4)

    print("INPUTS", inputs)
    print("LEN data test",len(data_test)-window)
    print(range(len(data_test)-window))
    for i in range(len(data_test)-window):
        print(i)
        p_test = model.predict(inputs)
        inputs = np.append(inputs,p_test)
        inputs = inputs.reshape(1, window+1, 4)
        inputs = inputs[0][-window:][:] 
        inputs = inputs.reshape(1, window, 4)
        output.append(data_test[i + window])
        output2.append(p_test)
        print(p_test)
        print(data_test[i + window])

    p_test = np.array(output2)
    output = np.array(output)
    p_test = p_test.reshape(len(data_test)-window,4)
    print(output.shape)
    print(output)

    p_test = scaler.inverse_transform(p_test)
    output = scaler.inverse_transform(output) 

    error_2 = calculate_error(output, p_test)
    error_2_rel = error_relativo(error_2, output)


    x_error = [x[0] for x in error_2]
    y_error = [x[1] for x in error_2]
    z_error = [x[2] for x in error_2]
    vx_error = [x[3] for x in error_2]
    vy_error = [x[4] for x in error_2]
    vz_error = [x[5] for x in error_2]

    fig, axs = plt.subplots(2, 3, sharex='col',figsize=(10, 6))
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.99, top=0.9, wspace=0.2, hspace=None)
    fig.suptitle('Coordenadas predichas vs reales y error cuadrático')

    axs[0,0].plot(output[:, 0], label="Valor real de x")
    axs[0,0].plot(p_test[:, 0], label="Predicción de x")
    axs[0,0].set_ylabel('x')
    axs[0,0].legend(loc='upper right')

    axs[0,1].plot(output[:, 1], label="Valor real de y")
    axs[0,1].plot(p_test[:, 1], label="Predicción de y")
    axs[0,1].set_ylabel('y')
    axs[0,1].legend(loc='upper right')

    axs[1,0].plot(x_error)
    axs[1,0].set_xlabel('TimeStep')
    axs[1,0].set_ylabel('x error cuadrático')

    axs[1,1].plot(y_error)
    axs[1,1].set_xlabel('TimeStep')
    axs[1,1].set_ylabel('y error cuadrático')

    plt.savefig('LSTM2D_best_Pred_coord_2.png', dpi=300)

    plt.show()


    fig, axs = plt.subplots(2, 3, sharex='col',figsize=(10, 6))
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.99, top=0.9, wspace=0.2, hspace=None)
    fig.suptitle('Velocidad de cada coordenada predichas vs reales y error cuadrático')

    axs[0,0].plot(output[:, 2], label="Valor real de vx")
    axs[0,0].plot(p_test[:, 2], label="Predicción de vx")
    axs[0,0].set_ylabel('vx')
    axs[0,0].legend(loc='upper right')

    axs[0,1].plot(output[:, 3], label="Valor real de vy")
    axs[0,1].plot(p_test[:, 3], label="Predicción de vy")
    axs[0,1].set_ylabel('vy')
    axs[0,1].legend(loc='upper right')

    axs[1,0].plot(vx_error)
    axs[1,0].set_ylabel('vx error cuadrático')
    axs[1,0].set_xlabel('TimeStep')

    axs[1,1].plot(vy_error)
    axs[1,1].set_ylabel('vy error cuadrático')
    axs[1,1].set_xlabel('TimeStep')

    plt.savefig('LSTM2D_best_Pred_vel_2.png', dpi=300)

    plt.show()"""

if __name__ == '__main__':
    main()
