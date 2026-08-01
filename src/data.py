import tensorflow as tf

from .config import BATCH_SIZE, DATASET_DIR, IMAGE_SIZE, SEED


AUTOTUNE = tf.data.AUTOTUNE


def normalize(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    if image.shape.rank == 3 and image.shape[-1] != 1:
        image = tf.image.rgb_to_grayscale(image)
    return image, label


def prepare_dataset(ds, training=False):
    ds = ds.map(normalize, num_parallel_calls=AUTOTUNE)
    if training:
        ds = ds.shuffle(4096, seed=SEED, reshuffle_each_iteration=True)
    return ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)


def load_english_mnist():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train[..., None]
    x_test = x_test[..., None]

    train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    train_size = int(x_train.shape[0] * 0.9)

    train = train.shuffle(x_train.shape[0], seed=SEED, reshuffle_each_iteration=False)
    val = train.skip(train_size)
    train = train.take(train_size)

    return prepare_dataset(train, True), prepare_dataset(val), prepare_dataset(test)


def load_devanagari_digits():
    train_dir = DATASET_DIR / "train"
    test_dir = DATASET_DIR / "test"
    if not train_dir.exists() or not test_dir.exists():
        raise FileNotFoundError(f"Expected Devanagari train/test folders under {DATASET_DIR}")

    kwargs = {
        "labels": "inferred",
        "label_mode": "int",
        "class_names": [f"digit_{i}" for i in range(10)],
        "color_mode": "grayscale",
        "image_size": IMAGE_SIZE,
        "batch_size": None,
        "seed": SEED,
    }
    train = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=0.1,
        subset="training",
        **kwargs,
    )
    val = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=0.1,
        subset="validation",
        **kwargs,
    )
    test = tf.keras.utils.image_dataset_from_directory(test_dir, **kwargs)

    return prepare_dataset(train, True), prepare_dataset(val), prepare_dataset(test)


def load_dataset(name):
    if name == "english":
        return load_english_mnist()
    if name == "devanagari":
        return load_devanagari_digits()
    raise ValueError("Dataset must be 'english' or 'devanagari'.")
