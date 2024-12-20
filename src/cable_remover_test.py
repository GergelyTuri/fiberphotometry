import os
import cv2
import glob
import numpy as np
from tqdm import tqdm

os.chdir("/mnt/DataDrive1/alex/Data_N_Test/cable_rmv_test")
os.makedirs("white_only", exist_ok=True)
mpg_files = glob.glob("*.mpg")

white_thresholds = np.arange(80, 95, 5)

for file in mpg_files:
    print(f"Processing {file}")
    cap = cv2.VideoCapture(file)

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writers = {}
    for thresh in white_thresholds:
        # Create VideoWriter object
        output_file = os.path.join(
            "white_only", f"{file.split('.')[0]}_threshold_{thresh}.avi"
        )
        writers[thresh] = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
        # mean_colors = []

    for _ in tqdm(range(total_frames), desc="Processing frames"):
        ret, frame = cap.read()
        if not ret:
            break

        for thresh in white_thresholds:
            white_mask = np.all(frame > thresh, axis=2)
            output_frame = np.zeros_like(frame)
            output_frame[white_mask] = frame[white_mask]
            writers[thresh].write(output_frame)

    # Release everything
    cap.release()
    for thresh in white_thresholds:
        writers[thresh].release()

print("Done")
print()
