
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense


def build_baseline_mlp(input_dim, learning_rate=0.001):
    """Build a baseline MLP for binary readmission prediction."""

    model = Sequential([
        Input(shape=(input_dim,)),

        Dense(64, activation="relu"),
        Dense(32, activation="relu"),

        Dense(1, activation="sigmoid")
    ], name="hospital_readmission_baseline_mlp")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate
        ),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="roc_auc"),
            tf.keras.metrics.AUC(
                curve="PR",
                name="pr_auc"
            )
        ]
    )

    return model


from tensorflow.keras.layers import Dropout
from tensorflow.keras.regularizers import l2


def build_regularized_mlp(
    input_dim,
    learning_rate=0.001,
    l2_strength=0.001,
    dropout_rate=0.20
):
    """Build a regularized MLP for binary readmission prediction."""

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),

        tf.keras.layers.Dense(
            64,
            activation="relu",
            kernel_regularizer=l2(l2_strength)
        ),
        Dropout(dropout_rate),

        tf.keras.layers.Dense(
            32,
            activation="relu",
            kernel_regularizer=l2(l2_strength)
        ),
        Dropout(dropout_rate),

        tf.keras.layers.Dense(
            1,
            activation="sigmoid"
        )
    ], name="hospital_readmission_regularized_mlp")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate
        ),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="roc_auc"),
            tf.keras.metrics.AUC(
                curve="PR",
                name="pr_auc"
            )
        ]
    )

    return model




def build_compact_mlp(
    input_dim,
    learning_rate=0.001,
    l2_strength=0.001,
    dropout_rate=0.20
):
    """Compact regularized ANN for binary readmission prediction."""

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),

        tf.keras.layers.Dense(
            32,
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(l2_strength)
        ),
        tf.keras.layers.Dropout(dropout_rate),

        tf.keras.layers.Dense(
            16,
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(l2_strength)
        ),
        tf.keras.layers.Dropout(dropout_rate),

        tf.keras.layers.Dense(
            1,
            activation="sigmoid"
        )
    ], name="hospital_readmission_compact_mlp")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate
        ),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="roc_auc"),
            tf.keras.metrics.AUC(
                curve="PR",
                name="pr_auc"
            )
        ]
    )

    return model


