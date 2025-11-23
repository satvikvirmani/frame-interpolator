import tensorflow as tf

def optimizer_with_cosine_annealing(train_dataset, INITIAL_LEARNING_RATE):
    lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=INITIAL_LEARNING_RATE,
        first_decay_steps=len(train_dataset) * 5,
        t_mul=2.0,
        m_mul=0.9,
        alpha=0.1
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    return optimizer