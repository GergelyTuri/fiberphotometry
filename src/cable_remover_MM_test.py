import os
import cv2
import glob
import numpy as np
from tqdm import tqdm

os.chdir("/mnt/DataDrive1/alex/Data_N_Test/cable_rmv_test")
os.makedirs("white_only", exist_ok=True)
os.makedirs("motion_only", exist_ok=True)
mpg_files = glob.glob("*.mpg")

white_threshold = None
motion_threshold = 10
kernel_size = (5, 5)
SIGMA = 1

cv2.setNumThreads(25)
FreezeThresh = 3500
MinDuration = 20


def blur_frame(frame: np.array) -> np.array:
    return cv2.GaussianBlur(frame.astype("float"), (0, 0), SIGMA)


def get_white_mask(frame: np.array, white_threshold: int) -> np.array:
    return np.all(frame > white_threshold, axis=2)


def get_non_white_mask(frame: np.array, white_threshold: int) -> np.array:
    return np.all(frame <= white_threshold, axis=2)


def remove_white(frame: np.array, white_threshold: int) -> np.array:
    # white_mask = get_white_mask(frame, white_threshold)
    non_white_mask = get_non_white_mask(frame, white_threshold)
    output_frame = np.zeros_like(frame)
    output_frame[non_white_mask] = frame[non_white_mask].astype("uint8")
    return output_frame


def calculate_freezing(Motion: list, FreezeThresh: int, MinDuration: int) -> np.array:
    # Find frames below thresh
    BelowThresh = (Motion < FreezeThresh).astype(int)

    # Perform local cumulative thresh detection
    # For each consecutive frame motion is below threshold count is increased by 1 until motion goes above thresh,
    # at which point coint is set back to 0
    CumThresh = np.zeros(len(Motion))
    for x in range(1, len(Motion)):
        if BelowThresh[x] == 1:
            CumThresh[x] = CumThresh[x - 1] + BelowThresh[x]

    # Define periods where motion is below thresh for minduration as freezing
    Freezing = (CumThresh >= MinDuration).astype(int)
    for x in range(len(Freezing) - 2, -1, -1):
        if Freezing[x] == 0 and Freezing[x + 1] > 0 and Freezing[x + 1] < MinDuration:
            Freezing[x] = Freezing[x + 1] + 1
    Freezing = (Freezing > 0).astype(int)
    Freezing = Freezing * 100  # Convert to Percentage

    return Freezing


for file in mpg_files:
    Motion = []
    print(f"Processing {file}")
    cap = cv2.VideoCapture(file)

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer4concatMovie = cv2.VideoWriter(
        f"motion_only/{file.split('.')[0]}.avi", fourcc, fps, (width * 2, height)
    )
    mean_color_val = np.inf

    pbar = tqdm(range(total_frames), desc="Finding minimum mean color value")
    for _ in pbar:
        ret, frame = cap.read()
        if not ret:
            break

        frame_blur = blur_frame(frame)

        curr_mean_color_val = np.mean(np.mean(frame, axis=(0, 1)))
        mean_color_val = np.minimum(mean_color_val, curr_mean_color_val)

        pbar.set_description(f"Current minimum: {mean_color_val}")

    white_threshold = np.ceil(mean_color_val).astype(int)
    print(f"White threshold set to: {white_threshold}")

    pbar2 = tqdm(range(1, total_frames), desc="Measuring motion")

    # reset video to the beginning
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    # get first frame
    ret, frame = cap.read()
    if not ret:
        continue

    frame = blur_frame(frame)

    for _ in pbar2:
        prev_frame = frame
        ret, frame = cap.read()
        if not ret:
            break

        frame = blur_frame(frame)

        frame_white_mask = get_white_mask(frame, white_threshold)
        prev_frame_white_mask = get_white_mask(prev_frame, white_threshold)
        combined_mask = ~(prev_frame_white_mask | frame_white_mask)

        frame_diff = np.abs(frame - prev_frame)
        frame_diff[combined_mask] = 0

        frame_cut = (frame_diff > motion_threshold).astype("uint8")
        Motion.append(np.sum(frame_cut))

    # reset video to the beginning again
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    Motion = np.array(Motion)
    Freezing = calculate_freezing(Motion, FreezeThresh, MinDuration)

    pbar3 = tqdm(range(total_frames), desc="Writing video")
    for frame_num in pbar3:
        ret, frame = cap.read()
        if not ret:
            break

        frame_noWhite = remove_white(blur_frame(frame), white_threshold)
        combined_frame = np.hstack((frame, frame_noWhite)).astype("uint8")
        if frame_num > len(Freezing) - 1:
            continue
        if Freezing[frame_num] > 0:
            text = "Freezing"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            color = (0, 0, 255)
            thickness = 2
            (text_width, text_height), _ = cv2.getTextSize(
                text, font, font_scale, thickness
            )
            x = width - text_width - 10
            y = text_height + 10
            cv2.putText(
                combined_frame, text, (x, y), font, font_scale, color, thickness
            )
        combined_frame = combined_frame.astype("uint8")
        writer4concatMovie.write(combined_frame)

    writer4concatMovie.release()
    cap.release()
    print()

print("Done")
print()
