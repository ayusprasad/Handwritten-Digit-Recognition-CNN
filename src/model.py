import tensorflow as tf

from .config import CHANNELS, IMAGE_SIZE, NUM_CLASSES


def conv_block(x, filters, dropout):
    x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    return tf.keras.layers.Dropout(dropout)(x)


def build_digit_cnn():
    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, CHANNELS))
    x = tf.keras.layers.RandomRotation(0.08)(inputs)
    x = tf.keras.layers.RandomTranslation(0.08, 0.08)(x)
    x = tf.keras.layers.RandomZoom(0.08)(x)

    x = conv_block(x, 32, 0.15)
    x = conv_block(x, 64, 0.20)
    x = conv_block(x, 128, 0.30)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Dropout(0.40)(x)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name="digit_cnn_batchnorm_dropout")
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=0.001, weight_decay=0.0001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
