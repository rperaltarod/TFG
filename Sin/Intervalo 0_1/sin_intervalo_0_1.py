import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
import os.path as path

# creamos el modelo
def create_model(punto_train):
    num_param = len(punto_train.keys())
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer([num_param], name='Input_Layer'),
        tf.keras.layers.Dense(35, activation='sigmoid', name='dense01'),
        tf.keras.layers.Dense(35, activation='sigmoid', name='dense02'),
        tf.keras.layers.Dense(35, activation='sigmoid', name='dense03'),
        tf.keras.layers.Dense(1, name='Output_Layer')
    ])

    learning_rate = 0.01
    model.compile(optimizer=tf.keras.optimizers.Adamax(learning_rate),
                  loss='mae', metrics=['mse', ])
    model.summary()
    return model


# creamos la funcion error cuadratico
def calculate_error(outcome_test, p_test):
    error = []
    for outs in zip(outcome_test, p_test):
        error.append((outs[0] - outs[1])**2)
    return error


def error_relativo(error, outcome_test):
    error_rel = np.sqrt(error) / np.absolute(outcome_test)
    return error_rel


def main():
    if path.exists('sin_0_1_data.pickle'):
        fichero = open('sin_0_1_data.pickle', 'rb')

        # Cargamos los datos del fichero
        X = pickle.load(fichero)
        Y = pickle.load(fichero)
        punto_train = pickle.load(fichero)
        punto_test = pickle.load(fichero)
        outcome_train = pickle.load(fichero)
        outcome_test = pickle.load(fichero)
        fichero.close()

    else:
        # elegimos numeros al azar
        X = np.arange(0, 1, 0.01)
        Y = np.arange(0, 1, 0.01)
        punto = pd.DataFrame(np.array(np.meshgrid(
            X, Y, )).T.reshape(-1, 2), columns=['X', 'Y'])

        print("punto", punto)
        # creamos las variables train y test
        punto_train = punto.sample(frac=0.8)
        punto_test = punto.drop(punto_train.index)

        print("punto_train", punto_train)
        print("punto_test", punto_test)

        outcome_train = np.sin(punto_train.X**2 + punto_train.Y**2)
        outcome_test = np.sin(punto_test.X**2 + punto_test.Y**2)

        print("outcome_train", outcome_train)
        print("outcome_test", outcome_test)

        pickled_file = open('sin_0_1_data.pickle', 'wb')

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
    fig1.savefig('sin_real_train_0_1.pdf')
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
    fig2.savefig('sin_real_test_0_1.pdf')
    plt.show()
    

    if path.exists('sin_0_1.h5'):

        model = tf.keras.models.load_model('sin_0_1.h5')
        fichero = open('history_model.pickle', 'rb')
        history = pickle.load(fichero)
        fichero.close()

    else:

        model = create_model(punto_train)
    
        # entrenamos el modelo
        history = model.fit(punto_train, outcome_train, batch_size=20,
                            epochs=120, validation_split=0.3, verbose=1)
        model.save('sin_0_1.h5')

        pickled_file = open('history_model.pickle', 'wb')

        pickle.dump(history.history, pickled_file)

        pickled_file.close()

    # evaluamos las perdidas
    loss, mae = model.evaluate(punto_test, outcome_test, verbose=2)
    print('Loss:', loss, 'MAE:', mae)

    # --------MONITOR
    # Plot training & validation loss values
    fig3 = plt.figure(figsize=(7, 6))
    fig3.tight_layout(pad=0.05)
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.ylabel('Pérdida')
    plt.xlabel('Épocas')
    plt.legend(['Entrenamiento', 'Validación'], loc='upper right')
    fig3.savefig('model_loss_sin_0_1.pdf')
    plt.show()

    # predecimos con los puntos de test
    p_test = model.predict(punto_test)

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
    fig4.savefig('sin_0_1_prediction_vs_real.pdf')
    plt.show()

    error = calculate_error(outcome_test, p_test)

    # Representamos el error relativo
    fig5 = plt.figure(figsize=(7, 6))
    plt.scatter(range(len(error)), error, s=2)
    fig5.tight_layout(pad=0.2)
    ax.set_ylabel('Error cuadrático', fontsize=12)
    ax.set_xlabel('Puntos (x,y)', fontsize=12)
    ax.tick_params(labelsize=10)
    fig5.savefig('error2D_0_1.pdf')
    plt.show()

    fig6 = plt.figure(figsize=(7, 6))
    ax = fig6.add_subplot(111, projection='3d')
    fig6.tight_layout(pad=0.2)
    ax.scatter(punto_test.X, punto_test.Y, error, s=2)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_zlabel('Error cuadrático', fontsize=12)
    ax.tick_params(labelsize=10)
    fig6.savefig('error3D_0_1.pdf')
    plt.show()  

    error=[item for sublist in error for item in sublist]
    print(len(error))
    outcome=np.array(outcome_test)  


if __name__ == '__main__':
    main()
