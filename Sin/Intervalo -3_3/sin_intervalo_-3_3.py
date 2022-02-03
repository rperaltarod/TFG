import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
import os.path as path

# creamos el modelo indicando el número de capas, el número de neuronas y la función de activación
def create_model(punto_train):
    num_param = len(punto_train.keys())
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer([num_param], name='Input_Layer'),
        tf.keras.layers.Dense(35, activation='sigmoid', name='dense01'),
        tf.keras.layers.Dense(35, activation='sigmoid', name='dense02'),
        tf.keras.layers.Dense(35, activation='sigmoid', name='dense03'),
        tf.keras.layers.Dense(1, name='Output_Layer')
    ])

    learning_rate = 0.01 #indicamos el ritmo de aprendizaje que se quiera aplicar

    #se compila el modelo con el optimizador, la función de pérdida y las métricas deseadas
    model.compile(optimizer=tf.keras.optimizers.Adamax(learning_rate),
                  loss='mae', metrics=['mse', ])
    model.summary()
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
    if path.exists('sin_-3_3_data.pickle'):
        fichero = open('sin_-3_3_data.pickle', 'rb')

        # Cargamos los datos del fichero de existir
        X = pickle.load(fichero)
        Y = pickle.load(fichero)
        punto_train = pickle.load(fichero)
        punto_test = pickle.load(fichero)
        outcome_train = pickle.load(fichero)
        outcome_test = pickle.load(fichero)
        fichero.close()

    else:
        #Creamos dos arrays de numeros entre -3 y 3 espaciados 0.01 de x e y
        X = np.arange(-3, 3, 0.01)
        Y = np.arange(-3, 3, 0.01)

        #Creamos una malla de puntos con todas las combinaciones de x e y
        punto = pd.DataFrame(np.array(np.meshgrid(
            X, Y, )).T.reshape(-1, 2), columns=['X', 'Y'])

        #Creamos las variables de entrada de train y test
        punto_train = punto.sample(frac=0.8)
        punto_test = punto.drop(punto_train.index)

        #creamos la salida obteniendo el valor de la función
        outcome_train = np.sin(punto_train.X**2 + punto_train.Y**2)
        outcome_test = np.sin(punto_test.X**2 + punto_test.Y**2)

        #guardamos las variables creada en un fichero 
        pickled_file = open('sin_-3_3_data.pickle', 'wb')

        pickle.dump(X, pickled_file)
        pickle.dump(Y, pickled_file)
        pickle.dump(punto_train, pickled_file)
        pickle.dump(punto_test, pickled_file)
        pickle.dump(outcome_train, pickled_file)
        pickle.dump(outcome_test, pickled_file)

        pickled_file.close()


    # representamos el seno obtenido con los valores de entrenamiento
    fig1 = plt.figure(figsize=(7, 6))
    ax = fig1.add_subplot(111, projection='3d')
    fig1.tight_layout(pad=0.05)
    ax.scatter(punto_train.X, punto_train.Y, outcome_train, s=2)
    ax.set_xlabel('x', fontsize=15)
    ax.set_ylabel('y', fontsize=15)
    ax.set_zlabel('z', fontsize=15)
    ax.tick_params(labelsize=10)
    fig1.savefig('sin_real_train_-3_3.png')
    plt.show()
    

    # representamos el seno obtenido con los valores de test
    fig2 = plt.figure(figsize=(7, 6))
    ax = fig2.add_subplot(111, projection='3d')
    fig2.tight_layout(pad=0.05)
    ax.scatter(punto_test.X, punto_test.Y, outcome_test, s=2)
    ax.set_xlabel('x', fontsize=15)
    ax.set_ylabel('y', fontsize=15)
    ax.set_zlabel('z', fontsize=15)
    ax.tick_params(labelsize=10)
    fig2.savefig('sin_real_test_-3_3.png')
    plt.show()
    

    if path.exists('sin_-3_3.h5'):

        model = tf.keras.models.load_model('sin_-3_3.h5')
        fichero = open('history_model.pickle', 'rb')
        history = pickle.load(fichero)
        fichero.close()

    else:

        model = create_model(punto_train)
    
        #Entrenamos el modelo con los puntos de entrada y salida de entrenamiento
        history = model.fit(punto_train, outcome_train, batch_size=40,
                            epochs=200, validation_split=0.25, verbose=1)
        history = history.history
        model.save('sin_-3_3.h5')

        pickled_file = open('history_model.pickle', 'wb')

        pickle.dump(history.history, pickled_file)

        pickled_file.close()

    #Evaluamos las perdidas
    loss, mae = model.evaluate(punto_test, outcome_test, verbose=2)
    print('Loss:', loss, 'MAE:', mae)

    #Representamos los valores de pérdida de entrenamiento y validación
    fig3 = plt.figure(figsize=(7, 6))
    fig3.tight_layout(pad=0.05)
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.ylabel('Pérdida')
    plt.xlabel('Épocas')
    plt.legend(['Entrenamiento', 'Validación'], loc='upper right')
    fig3.savefig('model_loss_sin_-3_3.pdf')
    plt.show()

    #Predecimos con los puntos de test
    p_test = model.predict(punto_test)

    #Representamos los valores de pérdida de entrenamiento y validación
    fig4 = plt.figure(figsize=(7, 6))
    ax = fig4.add_subplot(111, projection='3d')
    fig4.tight_layout(pad=0.05)
    ax.scatter(punto_test.X, punto_test.Y, outcome_test, label="Valor real", s=2)
    ax.scatter(punto_test.X, punto_test.Y, p_test, label="Valor predicho", s=2)
    ax.set_xlabel("x",fontsize=15)
    ax.set_ylabel("y",fontsize=15)
    ax.set_ylabel("z",fontsize=15)
    ax.tick_params(labelsize=10)
    plt.legend(loc='upper right')
    fig4.savefig('sin_-3_3_prediction_vs_real.png')
    plt.show()

    #Calculamos el error absoluto y el cuadrático de la predicción
    error_absoluto = calculate_error_absolute(outcome_test, p_test)
    error_cuad = calculate_error(outcome_test, p_test)

    #Representamos la predicción vs el valor real
    fig5 = plt.figure(figsize=(7, 6))
    plt.scatter(range(len(error_absoluto)), error_absoluto, s=2)
    fig5.tight_layout(pad=0.2)
    ax.set_ylabel('Error absoluto', fontsize=12)
    ax.set_xlabel('Puntos (x,y)', fontsize=12)
    ax.tick_params(labelsize=10)
    fig5.savefig('error2D_-3_3_error_abs.pdf')
    plt.show()

    #Representamos el error absoluto en 3D
    fig6 = plt.figure(figsize=(7, 6))
    ax = fig6.add_subplot(111, projection='3d')
    fig6.tight_layout(pad=0.2)
    ax.scatter(punto_test.X, punto_test.Y, error_absoluto, s=2)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_zlabel('Error absoluto', fontsize=12)
    ax.tick_params(labelsize=10)
    fig6.savefig('error3D_-3_3_error_abs.pdf')
    plt.show()


if __name__ == '__main__':
    main()
