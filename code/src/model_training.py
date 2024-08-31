import tensorflow as tf
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard

def build_and_train_model(x_train, y_train, x_test, y_test):
    """
    Builds and trains a neural network model using InceptionResNetV2 as the base.
    
    Args:
    x_train: Training data (images).
    y_train: Training labels (corresponding to the images).
    x_test: Testing data (images).
    y_test: Testing labels (corresponding to the images).

    Returns:
    Model: The trained Keras model.
    """
    # Define the base model using InceptionResNetV2 with pre-trained weights.
    # This model will extract features from the images but will not include its top classification layers.
    base_model = InceptionResNetV2(
        weights='imagenet',
        include_top=False,
        input_tensor=Input(shape=(224, 224, 3))  # Input layer for images of size 224x224 with 3 color channels (RGB).
    )
    
    # Add custom layers on top of the base model:
    # 1. Flatten the 3D feature maps into a 1D vector.
    # 2. Add fully connected (Dense) layers to process these features.
    # 3. Output layer with 4 units for classification, using sigmoid activation for multi-label tasks.
    x = base_model.output
    x = Flatten()(x)
    x = Dense(500, activation='relu')(x)
    x = Dense(250, activation='relu')(x)
    x = Dense(4, activation='sigmoid')(x)
    
    model = Model(inputs=base_model.input, outputs=x)
    
    # Compile the model with Mean Squared Error loss and Adam optimizer.
    # Adam optimizer adjusts the learning rate dynamically during training.
    model.compile(
        loss='mse',
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4)
    )
    
    model.summary()
    
    # Set up TensorBoard to monitor training progress and visualize metrics.
    tensorboard = TensorBoard(log_dir=r'..\output\logs')
    
    # Train the model with the provided training data and validate on the test data.
    # Training will run for 140 epochs, with TensorBoard logging progress.
    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=10,
        epochs=140,
        validation_data=(x_test, y_test),
        callbacks=[tensorboard]
    )
    
    return model
