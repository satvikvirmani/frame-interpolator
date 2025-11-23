import tensorflow as tf

vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
vgg.trainable = False
feature_extractor = tf.keras.Model(
    inputs=vgg.input,
    outputs=[
        vgg.get_layer('block2_conv2').output,
        vgg.get_layer('block3_conv3').output,
        vgg.get_layer('block4_conv3').output
    ]
)

def perceptual_loss(y_true, y_pred):
    y_true_resized = tf.image.resize(y_true, (224, 224))
    y_pred_resized = tf.image.resize(y_pred, (224, 224))

    y_true_vgg = tf.keras.applications.vgg19.preprocess_input(y_true_resized * 255.0)
    y_pred_vgg = tf.keras.applications.vgg19.preprocess_input(y_pred_resized * 255.0)

    true_features = feature_extractor(y_true_vgg)
    pred_features = feature_extractor(y_pred_vgg)

    loss = 0.0
    for t, p in zip(true_features, pred_features):
        loss += tf.reduce_mean(tf.abs(t - p))

    return loss / len(true_features)

def ssim_loss(y_true, y_pred):
    return 1.0 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))

def l1_loss(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true - y_pred))

def edge_loss(y_true, y_pred):
    def compute_edges(img):
        sobel_x = tf.constant([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=tf.float32)
        sobel_y = tf.constant([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=tf.float32)
        sobel_x = tf.reshape(sobel_x, [3, 3, 1, 1])
        sobel_y = tf.reshape(sobel_y, [3, 3, 1, 1])

        gray = tf.image.rgb_to_grayscale(img)

        edge_x = tf.nn.conv2d(gray, sobel_x, strides=[1, 1, 1, 1], padding='SAME')
        edge_y = tf.nn.conv2d(gray, sobel_y, strides=[1, 1, 1, 1], padding='SAME')

        return tf.sqrt(edge_x ** 2 + edge_y ** 2)

    true_edges = compute_edges(y_true)
    pred_edges = compute_edges(y_pred)

    return tf.reduce_mean(tf.abs(true_edges - pred_edges))

def combined_loss(y_true, y_pred):
    l1 = l1_loss(y_true, y_pred)
    ssim = ssim_loss(y_true, y_pred)
    perceptual = perceptual_loss(y_true, y_pred)
    edge = edge_loss(y_true, y_pred)

    total_loss = (
        0.3 * l1 +
        0.4 * ssim +
        0.2 * perceptual +
        0.1 * edge
    )

    return total_loss, l1, ssim, perceptual, edge