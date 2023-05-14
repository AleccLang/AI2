from keras.layers import Conv2D, MaxPooling2D, Dropout
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.utils import plot_model

def load_data():
    # Load data from .npy files
    X = np.load('Data/X.npy')
    Y = np.load('Data/Y.npy')

    print("Original Y shape:", Y.shape)  # Check original Y shape

    # Normalize X
    X = X / 255.0

    # Add a channel dimension to X
    X = np.expand_dims(X, axis=-1)

    # Convert Y to categorical (one-hot encoding) if necessary
    # Adjust the condition as necessary
    if len(Y.shape) == 1 or Y.shape[1] != 10:
        Y = to_categorical(Y)

    print("New Y shape:", Y.shape)  # Check new Y shape

    # Split data into training and validation sets
    X_train, X_val, Y_train, Y_val = train_test_split(
        X, Y, test_size=0.2, random_state=42)

    return X_train, X_val, Y_train, Y_val





"""def define_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform',
               padding='same', input_shape=(64, 64, 1)),
        MaxPooling2D((2, 2)),
        Dropout(0.4),
        Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform',
               padding='same'),
        MaxPooling2D((2, 2)),
        Dropout(0.4),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.6),
        Dense(10, activation='softmax')
    ])

    # Plot the model
    plot_model(model, to_file='model_plot.png',
               show_shapes=True, show_layer_names=True)

    return model"""


#second stage
"""def define_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform',
               padding='same', input_shape=(64, 64, 1)),
        MaxPooling2D((2, 2)),
        Dropout(0.4),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.6),
        Dense(10, activation='softmax')  # Assuming we have 10 classes (0-9)
    ])

    # Plot the model
    plot_model(model, to_file='model_plot.png',
               show_shapes=True, show_layer_names=True)

    return model"""

#baseline model
"""
def define_model():
    model = Sequential([
        Flatten(input_shape=(64, 64, 1)),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')  # Assuming we have 10 classes (0-9)
    ])

    # Plot the model
    plot_model(model, to_file='model_plot.png',
               show_shapes=True, show_layer_names=True)

    return model"""



def define_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform',
               padding='same', input_shape=(64, 64, 1)),
        MaxPooling2D((2, 2)),
        Dropout(0.4),
        Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform',
               padding='same'),
        MaxPooling2D((2, 2)),
        Dropout(0.5),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.6),
        Dense(10, activation='softmax')
    ])

    # Plot the model
    plot_model(model, to_file='model_plot.png',
               show_shapes=True, show_layer_names=True)

    return model


def compile_and_train_model(model, X_train, X_val, Y_train, Y_val):
    learning_rate = 0.0002
    batch_size = 64
    epochs = 500

    csv_logger = CSVLogger(f'training_log_lr={learning_rate}_batch={batch_size}_epochs={epochs}.csv',
                           append=True, separator=';')

    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(
        X_train,
        Y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, Y_val),
        callbacks=[csv_logger])  # Add this line
    return history


def plot_results(history, model_name):
    # Plot training & validation accuracy values
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.tight_layout()
    plt.savefig(f'{model_name}_plot.png')  # Saves the plot as a .png file
    plt.show()


def print_summary(history):
    # Get the number of epochs
    epochs = len(history.history['accuracy'])

    print(f"Training for {epochs} epochs.")

    print("\nFinal Training Results:")
    print(f"Accuracy: {history.history['accuracy'][-1]}")
    print(f"Loss: {history.history['loss'][-1]}")

    print("\nFinal Validation Results:")
    print(f"Accuracy: {history.history['val_accuracy'][-1]}")
    print(f"Loss: {history.history['val_loss'][-1]}")

    return {
        "epochs": epochs,
        "final_train_accuracy": history.history['accuracy'][-1],
        "final_train_loss": history.history['loss'][-1],
        "final_val_accuracy": history.history['val_accuracy'][-1],
        "final_val_loss": history.history['val_loss'][-1],
    }


def main():
    X_train, X_val, Y_train, Y_val = load_data()
    model = define_model()
    history = compile_and_train_model(model, X_train, X_val, Y_train, Y_val)
    plot_results(history, "Test2_0.0003_64_500")
    results = print_summary(history)


if __name__ == "__main__":
    main()
