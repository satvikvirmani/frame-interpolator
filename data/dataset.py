from data.preprocess import load_triplet, enhanced_augment
from data.triplet import process_videos
import tensorflow as tf
import os

class Dataset:
    def __init__(self, video_dir, output_dir, resize=(448, 256), frame_skip=4, group_stride=2, max_frames=None, test_ratio=0.2, img_quality=90, batch_size=16):
        self.video_dir = video_dir
        self.output_dir = output_dir
        self.resize = resize
        self.frame_skip = frame_skip
        self.group_stride = group_stride
        self.max_frames = max_frames
        self.test_ratio = test_ratio
        self.img_quality = img_quality
        self.batch_size = batch_size

        os.makedirs(output_dir, exist_ok=True)
        self.video_files = [f for f in os.listdir(video_dir) if f.endswith(".mp4")]
        
        self.train_file, self.test_file = process_videos(
            self.video_files,
            self.video_dir,
            self.output_dir,
            resize=self.resize,
            frame_skip=self.frame_skip,
            group_stride=self.group_stride,
            max_frames=self.max_frames,
            test_ratio=self.test_ratio,
            img_quality=self.img_quality
        )

    def create_dataset(self, list_file_path, is_training=True):
        with open(list_file_path, 'r') as f:
            path_suffixes = [line.strip() for line in f.readlines()]

        dataset = tf.data.Dataset.from_tensor_slices(path_suffixes)
        dataset = dataset.map(lambda x: tf.py_function(load_triplet, [x], [tf.float32, tf.float32]),
                            num_parallel_calls=tf.data.AUTOTUNE)

        if is_training:
            dataset = dataset.shuffle(1024)
            dataset = dataset.map(enhanced_augment, num_parallel_calls=tf.data.AUTOTUNE)

        dataset = dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        return dataset
    
    def get_datasets(self):
        train_dataset = self.create_dataset(self.train_file, is_training=True)
        val_dataset = self.create_dataset(self.test_file, is_training=False)
        return train_dataset, val_dataset