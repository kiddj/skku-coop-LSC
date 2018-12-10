from keras import layers, models
import matplotlib.pyplot as plt
from keras.models import load_model
from data_credit import load_credit
from plot_credit import plot_loss, plot_acc
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers
from pylab import rcParams

AE_modelPath = "models/ae_credit_16.h5"

[x_normal, x_anomaly] = load_credit()

print("Normal Shape: ", x_normal.shape)
print("Anomaly Shape: ", x_anomaly.shape)

RANDOM_SEED = 42

class AE(models.Model):
    def __init__(self, x_nodes, h1_dim, h2_dim):
        x = layers.Input(shape=(x_nodes, ), name='input')
        h1 = layers.Dense(h1_dim, activation='relu', activity_regularizer=regularizers.l1(10e-5), name='hidden1')(x)
        h2 = layers.Dense(h2_dim, activation='relu', name='hidden2')(h1)
        h3 = layers.Dense(h1_dim, activation='relu', name='hidden3')(h2)
        y = layers.Dense(x_nodes, activation='linear', name='output')(h3)

        super().__init__(x, y)

        self.x = x
        self.h1 = h1
        self.h2 = h2
        self.h1_dim = h1_dim
        self.h2_dim = h2_dim

        self.compile(optimizer='adam', loss='mean_squared_error',
                     metrics=['accuracy'])

    def get_encoder(self):
        encoder = models.Model(self.x, self.h2)
        # print("Encoder Info")
        # encoder.summary()
        return encoder

    def get_decoder(self):
        x = layers.Input(shape=(self.h2_dim, ))
        d1 = self.layers[-2](x)
        y = self.layers[-1](d1)
        decoder = models.Model(x, y)
        # print("Decoder Info")
        # decoder.summary()
        return decoder

    def save_encoder(self, filename):
        encoder = self.get_encoder()
        encoder.save(filename)
        print("[Saved] Encoder Info")
        encoder.summary()
        return

def ae_credit():
    x_nodes = 29
    h1_dim = 16
    h2_dim = 8

    ae = AE(x_nodes, h1_dim, h2_dim)

    x_normal_train, x_normal_valid = train_test_split(x_normal, test_size=0.2, random_state=RANDOM_SEED)

    _epochs = 30
    _batch_size = 32

    checkpointer = ModelCheckpoint(filepath=AE_modelPath, verbose=0, save_best_only=True)
    tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0,
                              write_graph=True, write_images=True)

    history = ae.fit(x_normal_train, x_normal_train, epochs=_epochs, batch_size=_batch_size,
                     shuffle=True, validation_data=(x_normal_valid, x_normal_valid),
                     verbose=1, callbacks=[checkpointer, tensorboard])

    ae.save_encoder("models/ae_credit_16_notbest.h5")

    plot_acc(history)
    plt.show()
    plot_loss(history)
    plt.show()

def encode_creidt(data_normal, data_anomaly, modelPath):
    encoder = load_model(modelPath)

    print("Load Encoder :: ", modelPath)
    encoder.summary()

    normal_encoded = encoder.predict(data_normal)
    anomaly_encoded = encoder.predict(data_anomaly)

    return [normal_encoded, anomaly_encoded]

if __name__ == '__main__':
    ae_credit()