import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import os.path as path
import pickle
from sklearn.preprocessing import MinMaxScaler
from tensorboard.plugins.hparams import api as hp
from datetime import datetime
from keras.models import Sequential
from keras.layers import Dense, InputLayer

#esta función crea el modelo con los hiperparámetros fijados
def create_model(hparams, capas, units, activ, optim, lrate):

    model = Sequential()
    model.add(InputLayer(1, name='Input_Layer'))
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

#esta función entrena el modelo y evalua las pérdidas
def train_test_model(model, in_train, out_train, in_test, out_test, x_val, y_val, logdir,
                     hparams, batch_s, epochs):

    history = model.fit(x=in_train,
                        y=out_train,
                        batch_size=hparams[batch_s],
                        epochs=hparams[epochs],
                        validation_data= (x_val, y_val), 
                        callbacks=[
                        tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1),  # log metrics
                        hp.KerasCallback(logdir, hparams),
                        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=200)],
                        verbose=1)


    # evaluamos las perdidas
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
    loss, mae = train_test_model(model, in_train, out_train, in_test, out_test, x_val, 
                                y_val, logdir, hparams, batch_s, epochs)
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


def main():

    if path.exists('model_standard2D_2_best.h5'): #si existe un modelo entrenado

        fichero = open('datos_standard2D.pickle', 'rb') #se abre el fichero con datos de entrada, salida, escaladores...

        # Cargamos los datos del fichero
        data_train = pickle.load(fichero)
        data_test = pickle.load(fichero)
        input_train = pickle.load(fichero)
        output_train = pickle.load(fichero)
        input_test = pickle.load(fichero)
        output_test = pickle.load(fichero)
        scaler = pickle.load(fichero)
        scaler2 = pickle.load(fichero)

        #se carga el modelo
        model = tf.keras.models.load_model('model_standard2D_2_best.h5')

    else: 

        filename_par = '../../fullTrajectories/1/parameters.csv' #direccion del fichero con los parametros de la trayectoria
        filename_traj = '../../fullTrajectories/1/trajectories.csv' #direccion del fichero con las coordenadas y velocidades de la trayectoria

        #se leen los parámetros de la trayectoria
        data_par = pd.read_csv(filename_par, names=["mu","Jacobi constant","Prop. time",
                                                    "x[0]","y[0]","vx[0]","vy[0]"])
        #se leen los coordenadas de la trayectoria
        data_traj = pd.read_csv(filename_traj, names=["TimeStep","x","y","vx","vy"])

        data = pd.DataFrame({"TimeStep": [], "x": [], "y": [],"vx": [],"vy": []})

        data_traj["TimeStep"] = list(np.arange(0,710,0.0071))


        data = data_traj.iloc[0:20000:1] #limitamos los datos totales a los 20000 primeros
        data_test = data.iloc[0:20000:2] #elegimos 10000 de test escogiendo 1 de cada 2
        data_train = data.drop(data_test.index) #los 10000 sobrantes quedan para testar

        scaler = MinMaxScaler(feature_range=(0,1)) #creamos un escalador entre un rango de 0 y 1

        #Ajustamos el escalador solo a los datos de entrenamiento
        scaler.fit(data_train[["x","y","vx","vy"]])

        
        data_train[["x","y","vx","vy"]] = scaler.transform(data_train[["x","y","vx","vy"]]) #se escalan los datos de entrenamiento de salida
        data_test[["x","y","vx","vy"]] = scaler.transform(data_test[["x","y","vx","vy"]]) #se escalan los datos de test de salida


        scaler2 = MinMaxScaler(feature_range=(0,1))
        scaler2.fit(data_train[["TimeStep"]])

        data_train[["TimeStep"]] = scaler2.transform(data_train[["TimeStep"]]) #se escalan los datos de test de entrada
        data_test[["TimeStep"]] = scaler2.transform(data_test[["TimeStep"]]) #se escalan los datos de test de entrada

        output_train = data_train[["x","y","vx","vy"]] #creamos la variable con la salida de entrenamiento
        input_train = data_train[["TimeStep"]] #creamos la variable con la entrada de entrenamiento

        output_test = data_test[["x","y","vx","vy"]] #creamos la variable con la salida de test
        input_test = data_test[["TimeStep"]] #creamos la variable con la entrada de test

        #
        input_train = input_train.reset_index(drop=True)
        input_test = input_test.reset_index(drop=True)

        output_train = output_train.reset_index(drop=True)
        output_test = output_test.reset_index(drop=True)


        fig = plt.figure()
        plt.scatter(output_train["x"] , output_train["y"], s=2, label= "Entrenamiento")
        plt.scatter(output_test["x"] , output_test["y"], s=2, label= "Test")
        #plt.title("Datos de entrenamiento y test de x e y")
        plt.xlabel("x",fontsize=15)
        plt.ylabel("y",fontsize=15)
        plt.yticks(fontsize=12)
        plt.xticks(fontsize=12)
        plt.legend(loc='upper right',fontsize=12)
        plt.savefig('standard2D_train_test_coord.pdf', dpi=300)
        plt.show()

        fig = plt.figure()
        plt.scatter(output_train["vx"] , output_train["vy"], s=2, label= "Entrenamiento")
        plt.scatter(output_test["vx"] , output_test["vy"], s=2, label= "Test")
        #plt.title("Datos de entrenamiento y test de vx e vy")
        plt.xlabel("vx",fontsize=15)
        plt.ylabel("vy",fontsize=15)
        plt.yticks(fontsize=12)
        plt.xticks(fontsize=12)
        plt.legend(loc='upper right',fontsize=12)
        plt.savefig('standard2D_train_test_vel.pdf', dpi=300)
        plt.show()

        X_val = input_train.iloc[0:len(input_train):5] #se dividen los datos de entrada de validación
        y_val = output_train.iloc[0:len(input_train):5] #se dividen los datos de salida de validación

        # se eliminan los datos de validación de los datos de entrenamiento
        input_train = input_train.drop(X_val.index) 
        output_train = output_train.drop(y_val.index)

        #se fijan los hiperparámetros que se usarán para los ensayos
        HP_NUM_CAPAS = hp.HParam("num_capas", hp.Discrete([9]))
        HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([300]))
        HP_ACTIVATION = hp.HParam('activation_function', hp.Discrete(['relu']))
        HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adamax']))
        HP_L_RATE = hp.HParam('lrate', hp.Discrete([0.001]))
        HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([100]))
        HP_EPOCHS = hp.HParam('epochs', hp.Discrete([20000]))

        METRIC_RMSE = 'rmse'
        METRIC_LOSS = 'loss'


        with tf.summary.create_file_writer('standard2D_2/logs/hparam_tuning').as_default():
          hp.hparams_config(
            hparams=[HP_NUM_CAPAS, HP_NUM_UNITS, HP_ACTIVATION, HP_OPTIMIZER, HP_L_RATE, HP_BATCH_SIZE, HP_EPOCHS],
            metrics=[hp.Metric(METRIC_RMSE, display_name='RMSE'),hp.Metric(METRIC_LOSS, display_name='Loss')],
          )

        # mediante los bucles anidades se llevan a cabo todos los ensayos producto de las combinaciones con los
        # hiperparámetros elegidos
        session_num = 300

        for num_capas in HP_NUM_CAPAS.domain.values:
            for num_units in HP_NUM_UNITS.domain.values:
                for activation_function in HP_ACTIVATION.domain.values:
                    for optimizer in HP_OPTIMIZER.domain.values:
                        for lrate in HP_L_RATE.domain.values:
                            for batch_size in HP_BATCH_SIZE.domain.values:
                                for epochs in HP_EPOCHS.domain.values:
                                    hparams = {
                                      HP_NUM_CAPAS: num_capas,
                                      HP_NUM_UNITS: num_units,
                                      HP_ACTIVATION: activation_function,
                                      HP_OPTIMIZER: optimizer,
                                      HP_L_RATE: lrate,
                                      HP_BATCH_SIZE: batch_size,
                                      HP_EPOCHS: epochs
                                    }

                                    log_dir = "standard2D_2/logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")

                                    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
                                    train_log_dir = 'standard2D_2/logs/gradient_tape/' + current_time + '/train'
                                    test_log_dir = 'standard2D_2/logs/gradient_tape/' + current_time + '/test'
                                    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
                                    test_summary_writer = tf.summary.create_file_writer(test_log_dir)

                                    logdir = "standard2D_2/logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")

                                    run_name = "run-%d" % session_num
                                    print('--- Starting trial: %s' % run_name)
                                    print({h.name: hparams[h] for h in hparams})
                                    model = run('standard2D_2/logs/hparam_tuning/' + run_name,
                                    hparams, HP_NUM_CAPAS, HP_NUM_UNITS, HP_ACTIVATION, HP_OPTIMIZER, HP_L_RATE, HP_BATCH_SIZE, HP_EPOCHS,
                                    input_train, output_train, input_test, output_test, X_val, y_val, logdir,METRIC_RMSE,METRIC_LOSS)

                                    session_num += 1

        pickled_file = open('datos_standard2D.pickle', 'wb')

        pickle.dump(data_train, pickled_file)
        pickle.dump(data_test, pickled_file)
        pickle.dump(input_train, pickled_file)
        pickle.dump(output_train, pickled_file)
        pickle.dump(input_test, pickled_file)
        pickle.dump(output_test, pickled_file)
        pickle.dump(scaler, pickled_file)
        pickle.dump(scaler2, pickled_file)

        pickled_file.close()

        model.save('model_standard2D_2_best.h5')


    # predecimos con los punto de test
    p_test = model.predict(input_test)

    #rescalamos los datos predichos y los reales para obtener los valores originales
    p_test = pd.DataFrame(scaler.inverse_transform(p_test), columns=output_test.columns)
    output_test = pd.DataFrame(scaler.inverse_transform(output_test), columns=output_test.columns)

    #se calcula el error cuadrático y el absoluto
    error_cuad = np.asarray(calculate_error(output_test.values, p_test.values))
    error_absoluto = np.asarray(calculate_error_absolute(output_test.values, p_test.values))

    fig = plt.figure()
    plt.scatter(output_test["x"] , output_test["y"], label="Valor real", s=2)
    plt.scatter(p_test["x"], p_test["y"], label="Predicción", s=2)
    #plt.title("Coordenadas reales vs predichas",fontsize=20)
    plt.xlabel("x",fontsize=15)
    plt.ylabel("y",fontsize=15)
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.legend(loc='upper right',fontsize=12)
    plt.savefig('standard2D_coord.pdf', dpi=200) 
    plt.show()

    fig = plt.figure()
    plt.scatter(output_test["vx"] , output_test["vy"],label="Valor real", s=2)
    plt.scatter(p_test["vx"], p_test["vy"],label="Predicción", s=2)
    plt.xlabel("vx",fontsize=15)
    plt.ylabel("vy",fontsize=15)
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.legend(loc='upper right',fontsize=12)
    plt.savefig('standard2D_vel.pdf', dpi=200) 
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

    plt.savefig('standard2D_coord_error_abs.pdf', dpi=200)

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

    plt.savefig('standard2D_vel_error_abs.pdf', dpi=200)
    plt.show()

if __name__ == '__main__':
    main()
