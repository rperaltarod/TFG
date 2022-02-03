import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
import os.path as path
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from tensorboard.plugins.hparams import api as hp
from keras.models import Sequential
from keras.layers import Dense, InputLayer

def scaler_data(columns,data_train, data_test):

    scaler = MinMaxScaler(feature_range=(0, 1))

    # Ajustamos solo a los datos de entrenamiento.
    scaler.fit(data_train[columns])

    # Aplicamos la transformación tantos al conjunto de entrenamiento como al de test.
    data_train[columns] = scaler.transform(
        data_train[columns])
    data_test[columns] = scaler.transform(
        data_test[columns])

    return scaler, data_train, data_test

def create_model(hparams, capas, units, activ, optim, lrate):

    model = Sequential()
    model.add(InputLayer(6, name='Input_Layer'))
    for i in range(hparams[capas]):
        model.add(Dense(hparams[units], activation=hparams[activ], name=f'dense_{i}'))
    model.add(Dense(4, name='Output_Layer'))

    optimizer_name = hparams[optim]
    learning_rate = hparams[lrate]
    if optimizer_name == "adamax":
        optimizer = tf.keras.optimizers.Adamax(learning_rate=learning_rate)
    elif optimizer_name == "sgd":
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    else:
        raise ValueError("unexpected optimizer name: %r" % (optimizer_name,))
    model.compile(optimizer=hparams[optim],
                loss='mse',
                metrics=["mae"])

    model.summary()
    return model

def train_test_model(model, in_train, out_train, in_test, out_test, x_val, y_val, logdir, hparams, batch_s, epochs):

    history = model.fit(x=in_train,
                        y=out_train,
                        batch_size=hparams[batch_s],
                        epochs=hparams[epochs],
                        validation_data= (x_val, y_val), 
                        callbacks=[
                        tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1),  # log metrics
                        hp.KerasCallback(logdir, hparams),
                        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)],
                        verbose=1)

    #Evaluamos las perdidas
    loss, mae= model.evaluate(x=in_test, y=out_test, verbose=2)
    print('Loss:', loss, 'MAE:', mae)

    #Representamos los valores de pérdida de entrenamiento y validación
    fig1 = plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validate'], loc='upper right')
    plt.savefig('standard2D_best_Loss.png', dpi=300)

    #Representamos los valores del mae de entrenamiento y validación
    fig2 = plt.figure(figsize=(10, 6))
    plt.plot(history.history['mae'])
    plt.plot(history.history['val_mae'])
    plt.title('Model mae')
    plt.ylabel('MAE')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validate'], loc='upper right')
    plt.savefig('standard2D_best_MAE.png', dpi=300)

    return loss, mae

#esta funcion llama a las funciones que crean el modelo y lo entrenan
def run(run_dir, hparams, capas, units, activ ,optim, lrate, batch_s, epochs, in_train,
        out_train, in_test, out_test, x_val, y_val, logdir, METRIC_MAE, METRIC_LOSS):
  with tf.summary.create_file_writer(run_dir).as_default():

    hp.hparams(hparams)  # record the values used in this trial
    model = create_model(hparams, capas, units, activ, optim, lrate)
    loss, mae = train_test_model(model, in_train, out_train, in_test, out_test, x_val, y_val, logdir, hparams, batch_s, epochs)
    tf.summary.scalar(METRIC_MAE, mae, step=1)
    tf.summary.scalar(METRIC_LOSS, loss, step=1)
    return model

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

# creamos la funcion para calcular el error relativo
def error_relativo(error, outcome_test):
    error_rel = np.sqrt(error) / np.absolute(outcome_test)
    return error_rel

def main():
    #si existe un modelo entrenado
    if path.exists('model_standard2D_multiple_best.h5'):

        #cargamos el modelo guardado
        model = tf.keras.models.load_model('model_standard2D_multiple_best.h5')

        fichero = open('datos_standard2D_multiple.pickle', 'rb')

        # Cargamos los datos del fichero
        input_train = pickle.load(fichero)
        input_test = pickle.load(fichero)
        output_train = pickle.load(fichero)
        output_test = pickle.load(fichero)
        scaler1 = pickle.load(fichero)
        scaler2 = pickle.load(fichero)

    else:

        input_train = pd.DataFrame({"mu": [], "Jacobi constant": [], "Prop. time": [], "x[0]": [
        ], "y[0]": [], "vx[0]": [], "vy[0]": []})

        input_test = pd.DataFrame({"mu": [], "Jacobi constant": [], "Prop. time": [], "x[0]": [
        ], "y[0]": [], "vx[0]": [], "vy[0]": []})


        filenames_par_train = ('../../fullTrajectories/1/parameters.csv','../../fullTrajectories/2/parameters.csv',
                               '../../fullTrajectories/3/parameters.csv')
        filenames_par_test = ('../../fullTrajectories/4/parameters.csv')

        filenames_traj_train = ('../../fullTrajectories/1/trajectories.csv','../../fullTrajectories/2/trajectories.csv',
                               '../../fullTrajectories/3/trajectories.csv')
        filenames_traj_test = ('../../fullTrajectories/4/trajectories.csv')

        #cargamos los parámetros y las coordenadas y velocidades de cada trayectoria
        
        for i in range(10000):
            input_test = input_test.append(pd.read_csv(filenames_par_test, names=["mu", "Jacobi constant",
                                                                            "Prop. time", "x[0]", "y[0]", "vx[0]", "vy[0]"]))
            
        input_test = input_test.reset_index(drop=True)

        for f in filenames_par_train:
            for i in range(10000):
                input_train = input_train.append(pd.read_csv(f, names=["mu", "Jacobi constant",
                                                                            "Prop. time", "x[0]", "y[0]", "vx[0]", "vy[0]"]))

        input_train = input_train.reset_index(drop=True)

        time_step_train = pd.DataFrame({"TimeStep":[]})
        time_step_test = pd.DataFrame({"TimeStep":[]})   

        output_train = pd.DataFrame({"x":[], "y":[], "vx": [], "vy": []})
        output_test = pd.DataFrame({"x":[], "y":[], "vx": [], "vy": []})

        for f in filenames_traj_train:
            data_traj_train = pd.read_csv(f, names=["TimeStep", "x", "y", "vx", "vy"])
            data_traj_train["TimeStep"] = list(np.arange(0,710,0.0071))

            data = data_traj_train.iloc[0:10000:1] #se seleccionan los 10000 primero puntos de cada trayectoria
            output_train = output_train.append(data[["x", "y", "vx", "vy"]]) 
            time_step_train = time_step_train.append(data[["TimeStep"]])
        
        data_traj_test = pd.read_csv(filenames_traj_test, names=["TimeStep", "x", "y", "vx", "vy"])
        data_traj_test["TimeStep"] = list(np.arange(0,710,0.0071))

        data = data_traj_test.iloc[0:10000:1] 
        output_test = data[["x", "y", "vx", "vy"]]
        time_step_test = data[["TimeStep"]]

        time_step_train = time_step_train.reset_index(drop=True)
        time_step_test = time_step_test.reset_index(drop=True)


        input_train.loc[:, 'TimeStep'] = time_step_train
        input_test.loc[:, 'TimeStep'] = time_step_test

        for i in range(0,len(output_test["x"]),10000):
            fig = plt.figure(figsize=(9,6))
            plt.scatter(output_test.iloc[i:10000+i]["x"] , output_test.iloc[i:10000+i]["y"], s=2, c= "orange")
            plt.xlabel("x",fontsize=15)
            plt.ylabel("y",fontsize=15)
            plt.yticks(fontsize=12)
            plt.xticks(fontsize=12)
            plt.savefig(f'multiple_standard2D_coord_test{i}.pdf', dpi=300) 
            plt.show()

            fig = plt.figure(figsize=(9,6))
            plt.scatter(output_test.iloc[i:10000+i]["vx"] , output_test.iloc[i:10000+i]["vy"], s=2, c= "orange")
            plt.xlabel("vx",fontsize=15)
            plt.ylabel("vy",fontsize=15)
            plt.yticks(fontsize=12)
            plt.xticks(fontsize=12)

            plt.savefig(f'multiple_standard2D_vel_test{i}.pdf', dpi=300) 
            plt.show()

        for i in range(0,len(output_train["x"]),10000):
            fig = plt.figure(figsize=(9,6))
            plt.scatter(output_train.iloc[i:10000+i]["x"] , output_train.iloc[i:10000+i]["y"], s=2)
            plt.xlabel("x",fontsize=15)
            plt.ylabel("y",fontsize=15)
            plt.yticks(fontsize=12)
            plt.xticks(fontsize=12)
            plt.savefig(f'multiple_standard2D_coord_train{i}.pdf', dpi=300) 
            plt.show()

            fig = plt.figure(figsize=(9,6))
            plt.scatter(output_train.iloc[i:10000+i]["vx"] , output_train.iloc[i:10000+i]["vy"], s=2)
            plt.xlabel("vx",fontsize=15)
            plt.ylabel("vy",fontsize=15)
            plt.yticks(fontsize=12)
            plt.xticks(fontsize=12)
            plt.savefig(f'multiple_standard2D_vel_train{i}.pdf', dpi=300) 
            plt.show()


        input_train = input_train.drop(columns=["mu", "Prop. time"])
        input_test = input_test.drop(columns=["mu", "Prop. time"])

        #escalamos los datos
        scaler1, input_train, input_test = scaler_data(["Jacobi constant", "x[0]", "y[0]", "vx[0]", "vy[0]"], input_train, input_test)
        scaler2, output_train, output_test = scaler_data(["x","y","vx","vy"], output_train, output_test)

        #seleccionamos los datos de validación
        X_val = input_train.iloc[0:input_train.shape[0]:4]
        y_val = output_train.iloc[0:output_train.shape[0]:4]

        input_train = input_train.drop(index=X_val.index)
        output_train = output_train.drop(index=y_val.index)

        pickled_file = open('datos_standard2D_multiple.pickle', 'wb')

        pickle.dump(input_train, pickled_file)
        pickle.dump(input_test, pickled_file)
        pickle.dump(output_train, pickled_file)
        pickle.dump(output_test, pickled_file)
        pickle.dump(scaler1, pickled_file)
        pickle.dump(scaler2, pickled_file)

        pickled_file.close()

        #se seleccionan aquellos hiperparámetros y las métricas con los que se quieran ensayar
        HP_NUM_CAPAS = hp.HParam('num_capas', hp.Discrete([3]))
        HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([300]))
        HP_ACTIVATION = hp.HParam('activation_function', hp.Discrete(['relu']))
        HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adamax']))
        HP_L_RATE = hp.HParam('lrate', hp.Discrete([0.001,]))
        HP_BATCH_SIZE = hp.HParam('batch_s', hp.Discrete([200]))
        HP_EPOCHS = hp.HParam('epochs', hp.Discrete([10000]))

        METRIC_MAE = 'mae'
        METRIC_LOSS = 'loss'

        with tf.summary.create_file_writer('standard_multiple/logs/hparam_tuning').as_default():
          hp.hparams_config(
            hparams=[HP_NUM_CAPAS, HP_NUM_UNITS, HP_ACTIVATION, HP_OPTIMIZER, HP_L_RATE, HP_BATCH_SIZE, HP_EPOCHS],
            metrics=[hp.Metric(METRIC_MAE, display_name='MAE'),hp.Metric(METRIC_LOSS, display_name='Loss')],
          )

        # mediante los bucles anidades se llevan a cabo todos los ensayos producto de las combinaciones con los
        # hiperparámetros elegidos
        session_num = 60
        for num_capas in HP_NUM_CAPAS.domain.values:
            for num_units in HP_NUM_UNITS.domain.values:
                for activation_function in HP_ACTIVATION.domain.values:
                    for optimizer in HP_OPTIMIZER.domain.values:
                        for lrate in HP_L_RATE.domain.values:
                            for batch_s in HP_BATCH_SIZE.domain.values:
                                for epochs in HP_EPOCHS.domain.values:
                                        hparams = {
                                          HP_NUM_CAPAS: num_capas,
                                          HP_NUM_UNITS: num_units,
                                          HP_ACTIVATION: activation_function,
                                          HP_OPTIMIZER: optimizer,
                                          HP_L_RATE: lrate,
                                          HP_BATCH_SIZE: batch_s,
                                          HP_EPOCHS: epochs,
                                        }

                                        log_dir = "standard_multiple/logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")

                                        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
                                        train_log_dir = 'standard_multiple/logs/gradient_tape/' + current_time + '/train'
                                        test_log_dir = 'standard_multiple/logs/gradient_tape/' + current_time + '/test'
                                        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
                                        test_summary_writer = tf.summary.create_file_writer(test_log_dir)

                                        logdir = "standard_multiple/logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")

                                        print("HP_NUM_UNITS:",hparams)
                                        run_name = "run-%d" % session_num
                                        print('--- Starting trial: %s' % run_name)
                                        print({h.name: hparams[h] for h in hparams})
                                        model = run('standard_multiple/logs/hparam_tuning/' + run_name,
                                        hparams, HP_NUM_CAPAS, HP_NUM_UNITS, HP_ACTIVATION, HP_OPTIMIZER, HP_L_RATE, HP_BATCH_SIZE, HP_EPOCHS,
                                        input_train, output_train, input_test, output_test, X_val, y_val, logdir,METRIC_MAE,METRIC_LOSS)


                                        session_num += 1

        model.save('model_standard2D_multiple_best.h5')

    # predecimos con los puntos de test
    p_test = model.predict(input_test)
    
    output_test = pd.DataFrame(scaler2.inverse_transform(output_test), columns=output_test.columns)
    p_test = pd.DataFrame(scaler2.inverse_transform(p_test), columns=output_test.columns)

    error = np.asarray(calculate_error(output_test.values, p_test.values))
    error_absoluto = np.asarray(calculate_error_absolute(output_test.values, p_test.values))
    error_rel = error_relativo(error, output_test)


    fig = plt.figure()
    plt.scatter(output_test["x"] , output_test["y"], label="Valor real", s=2)
    plt.scatter(p_test["x"], p_test["y"], label="Predicción", s=2)
    plt.xlabel("x",fontsize=15)
    plt.ylabel("y",fontsize=15)
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.legend(loc='upper right',fontsize=12)
    plt.savefig('multiple_standard2D_coord.pdf', dpi=300) 
    plt.show()

    fig = plt.figure()
    plt.scatter(output_test["vx"] , output_test["vy"],label="Valor real", s=2)
    plt.scatter(p_test["vx"], p_test["vy"],label="Predicción", s=2)
    plt.xlabel("vx",fontsize=15)
    plt.ylabel("vy",fontsize=15)
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.legend(loc='upper right',fontsize=12)
    plt.savefig('multiple_standard2D_vel.pdf', dpi=300) 
    plt.show()


    fig, axs = plt.subplots(2, 2, sharex='col',figsize=(10, 6))
    plt.subplots_adjust(left=0.08, bottom=0.1, right=0.96, top=0.9, wspace=0.2, hspace=None)
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

    plt.savefig('multiple_standard2D_error_abs_Pred_coord.png', dpi=300)

    plt.show()

    fig, axs = plt.subplots(2, 2, sharex='col',figsize=(10, 6))
    plt.subplots_adjust(left=0.08, bottom=0.1, right=0.96, top=0.9, wspace=0.2, hspace=None)
    #fig.suptitle('Velocidad de cada coordenada predichas vs reales y error cuadrático',fontsize=20)

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

    plt.savefig('multiple_standard2D_error_abs_Pred_vel.png', dpi=300)
    plt.show()

if __name__ == '__main__':
    main()
