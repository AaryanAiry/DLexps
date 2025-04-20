import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler

# Load and scale the Iris dataset
iris = load_iris()
X = iris.data
X = MinMaxScaler().fit_transform(X)  # scale to [0,1]

# Build the autoencoder
input_dim = X.shape[1]

# Encoder: 4 -> 2
# Decoder: 2 -> 4
autoencoder = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(input_dim,)),
    tf.keras.layers.Dense(2, activation='relu'),       # Encoder
    tf.keras.layers.Dense(input_dim, activation='sigmoid')  # Decoder
])

# Compile and train
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(X, X, epochs=100, verbose=0)

# Encode and decode
encoded = autoencoder.layers[0](X)
decoded = autoencoder.predict(X)

print("Original:", X[0])
print("Decoded :", decoded[0])

