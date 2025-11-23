from tqdm.notebook import tqdm
import random as random
from math import floor
import os as os
import time as time
import cv2 as cv2

def extract_triplets_from_video(video_path, output_root, resize=(448, 256), frame_skip=1, group_stride=2, max_frames=None, img_quality=90):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return []

    os.makedirs(output_root, exist_ok=True)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames:
        total_frames = min(total_frames, max_frames)

    print(f"\nProcessing: {os.path.basename(video_path)}  |  Total frames: {total_frames}")
    triplet_dirs = []

    frame_buffer = []
    frame_index = 0

    with tqdm(total=total_frames, desc="Extracting frames", unit="f") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret or frame_index >= total_frames:
                break

            if frame_index % frame_skip == 0:
                frame_resized = cv2.resize(frame, resize)
                frame_buffer.append(frame_resized)

                if len(frame_buffer) > 3:
                    frame_buffer.pop(0)

                if len(frame_buffer) == 3 and frame_index % group_stride == 0:
                    triplet_id = f"{len(triplet_dirs):04d}"
                    triplet_dir = os.path.join(output_root, triplet_id)
                    os.makedirs(triplet_dir, exist_ok=True)

                    for j, img in enumerate(frame_buffer):
                        cv2.imwrite(os.path.join(triplet_dir, f"img{j}.jpg"), img,
                                    [cv2.IMWRITE_JPEG_QUALITY, img_quality])

                    triplet_dirs.append(triplet_dir)

            frame_index += 1
            pbar.update(1)

    cap.release()
    return triplet_dirs

def process_videos(video_files, video_dir, output_dir, resize=(448, 256), frame_skip=4, group_stride=2, max_frames=None, test_ratio=0.2, img_quality=90):
    all_triplets = []

    for video_file in tqdm(video_files, desc="Processing videos", unit="vid"):
        video_path = os.path.join(video_dir, video_file)
        video_name = os.path.splitext(video_file)[0]
        output_subdir = os.path.join(output_dir, video_name)

        triplet_paths = extract_triplets_from_video(
            video_path,
            output_subdir,
            resize=resize,
            frame_skip=frame_skip,
            group_stride=group_stride,
            max_frames=max_frames,
            img_quality=img_quality,
        )
        all_triplets.extend(triplet_paths)

    all_triplets_rel = [os.path.relpath(p, start=output_dir) for p in all_triplets]

    random.shuffle(all_triplets_rel)
    split_index = floor(len(all_triplets_rel) * (1 - test_ratio))
    train_list = all_triplets_rel[:split_index]
    test_list = all_triplets_rel[split_index:]

    train_txt = os.path.join(output_dir, "train.txt")
    test_txt = os.path.join(output_dir, "test.txt")

    with open(train_txt, "w") as f:
        f.write("\n".join(train_list))
    with open(test_txt, "w") as f:
        f.write("\n".join(test_list))

    return train_txt, test_txt