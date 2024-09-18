import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# Load sample dataset (Iris dataset)
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)

# Create a simple model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(10, input_dim=4, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=5)

# Save model
model.save('model.h5')
