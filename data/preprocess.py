import tensorflow as tf
import os

def load_and_preprocess_image(path, img_height=256, img_width=448):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [img_height, img_width])
    return image

def load_triplet(path_suffix, img_height=256, img_width=448, data_dir='data/dataset'):
    path_suffix_str = path_suffix.numpy().decode('utf-8')
    base_path = os.path.join(data_dir, path_suffix_str)

    path0 = os.path.join(base_path, 'img0.jpg')
    path1 = os.path.join(base_path, 'img1.jpg')
    path2 = os.path.join(base_path, 'img2.jpg')

    frame0 = load_and_preprocess_image(path0)
    frame1_target = load_and_preprocess_image(path1)
    frame2 = load_and_preprocess_image(path2)

    input_frames = tf.concat([frame0, frame2], axis=-1)
    return input_frames, frame1_target

def enhanced_augment(input_frames, target):
    frame1 = input_frames[..., :3]
    frame2 = input_frames[..., 3:]

    if tf.random.uniform(()) > 0.5:
        frame1 = tf.image.flip_left_right(frame1)
        frame2 = tf.image.flip_left_right(frame2)
        target = tf.image.flip_left_right(target)

    brightness_delta = tf.random.uniform((), -0.15, 0.15)
    frame1 = tf.image.adjust_brightness(frame1, brightness_delta)
    frame2 = tf.image.adjust_brightness(frame2, brightness_delta)
    target = tf.image.adjust_brightness(target, brightness_delta)

    if tf.random.uniform(()) > 0.5:
        contrast_factor = tf.random.uniform((), 0.8, 1.2)
        frame1 = tf.image.adjust_contrast(frame1, contrast_factor)
        frame2 = tf.image.adjust_contrast(frame2, contrast_factor)
        target = tf.image.adjust_contrast(target, contrast_factor)

    if tf.random.uniform(()) > 0.5:
        saturation_factor = tf.random.uniform((), 0.8, 1.2)
        frame1 = tf.image.adjust_saturation(frame1, saturation_factor)
        frame2 = tf.image.adjust_saturation(frame2, saturation_factor)
        target = tf.image.adjust_saturation(target, saturation_factor)

    frame1 = tf.clip_by_value(frame1, 0.0, 1.0)
    frame2 = tf.clip_by_value(frame2, 0.0, 1.0)
    target = tf.clip_by_value(target, 0.0, 1.0)

    input_frames = tf.concat([frame1, frame2], axis=-1)

    return input_frames, target