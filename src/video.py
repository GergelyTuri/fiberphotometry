"""
Routine to handle video files.
by: Gergely Turi
gt2253@cumc.columbia.edu
gt2253@cumc.columbia.edu
"""

from dataclasses import dataclass
import os
from os.path import join
from enum import Enum
from .FreezeAnalysis import create_video_dict


from cv2 import (
    CAP_PROP_FPS,
    CAP_PROP_FRAME_COUNT,
    CAP_PROP_FRAME_HEIGHT,
    CAP_PROP_FRAME_WIDTH,
    CAP_PROP_POS_FRAMES,
    VideoCapture,
    VideoWriter,
    VideoWriter_fourcc,
)


@dataclass
class UsbVideo:
    """Video class to handle video files recored with USB camera.
    Initialize with the path to the video file.

    Parameters:
    -----------
    vid_path: str
        Path to the video file.
    dsmpl: int
        Downsampling factor.
    width: int
        Width of the video.
    height: int
        Height of the video.
    start: int
        Start frame.
    end: int
        End frame.
    preCue_onset: int
        Frames before cue onset. If None, will be calculated as seconds4Cue * fps.
        If not None, needs to be an integer that represents total frames.
    postCue_onset: int
        Frames after cue onset. If None, will be calculated as seconds4Cue * fps.
        If not None, needs to be an integer that represents total frames.
    seconds4Cue: int
        Seconds before and after cue onset. If preCue_onset and postCue_onset are
        not None, will be ignored.


    Parameters:
    -----------
    vid_path: str
        Path to the video file.
    dsmpl: int
        Downsampling factor.
    width: int
        Width of the video.
    height: int
        Height of the video.
    start: int
        Start frame.
    end: int
        End frame.
    preCue_onset: int
        Frames before cue onset. If None, will be calculated as seconds4Cue * fps.
        If not None, needs to be an integer that represents total frames.
    postCue_onset: int
        Frames after cue onset. If None, will be calculated as seconds4Cue * fps.
        If not None, needs to be an integer that represents total frames.
    seconds4Cue: int
        Seconds before and after cue onset. If preCue_onset and postCue_onset are
        not None, will be ignored.

    Example:
    >>> video = UsbVideo(vid_path="path/to/video.avi")"""

    vid_path: str
    dsmpl: int = 1
    width: int = 1
    height: int = 1
    start: int = 0
    end: int = None
    preCue_onset: int = None
    postCue_onset: int = None
    seconds4Cue: int = 30
    dsmpl: int = 1
    width: int = 1
    height: int = 1
    start: int = 0
    end: int = None
    preCue_onset: int = None
    postCue_onset: int = None
    seconds4Cue: int = 30

    def __post_init__(self):
        """Reads in all the data from tdt recording.
        f_path: str
            Path to the folder containing data."""
        self.video = VideoCapture(self.vid_path)
        self.sessFolder = os.path.dirname(self.vid_path)
        self.sessFolder = os.path.dirname(self.vid_path)

    @property
    def video_params(self) -> dict:
        """Returns the parameters of the recorded video."""
        params = {}
        cap = self.video
        if cap.isOpened():
            # get cap property
            params["width"] = cap.get(CAP_PROP_FRAME_WIDTH)  # float `width`
            params["height"] = cap.get(CAP_PROP_FRAME_HEIGHT)  # float `height`
            params["fps"] = cap.get(CAP_PROP_FPS)
            params["frame_count"] = cap.get(CAP_PROP_FRAME_COUNT)
        cap.release()

        self.fps = int(params["fps"])

        if self.preCue_onset is not None and self.postCue_onset is not None:
            self.seconds4Cue = None
        else:
            self.preCue_onset = (
                int(self.fps * self.seconds4Cue)
                if self.preCue_onset is None
                else self.preCue_onset
            )
            self.postCue_onset = (
                int(self.fps * self.seconds4Cue)
                if self.postCue_onset is None
                else self.postCue_onset
            )


        self.fps = int(params["fps"])

        if self.preCue_onset is not None and self.postCue_onset is not None:
            self.seconds4Cue = None
        else:
            self.preCue_onset = (
                int(self.fps * self.seconds4Cue)
                if self.preCue_onset is None
                else self.preCue_onset
            )
            self.postCue_onset = (
                int(self.fps * self.seconds4Cue)
                if self.postCue_onset is None
                else self.postCue_onset
            )

        return params

    @property
    def video_create4FreezeAnalysis(self) -> dict:
        return create_video_dict(
            video_path=self.vid_path,
            start=self.start,
            end=self.end,
            dsmpl=self.dsmpl,
            width=self.width,
            height=self.height,
            fps=self.fps,
        )

    @property
    def video_create4FreezeAnalysis(self) -> dict:
        return create_video_dict(
            video_path=self.vid_path,
            start=self.start,
            end=self.end,
            dsmpl=self.dsmpl,
            width=self.width,
            height=self.height,
            fps=self.fps,
        )

    def slice_video(self, start: int, end: int, codec: str) -> None:
        """Slices the video between start and end frames.
        Parameters:
        -----------
        start: int
            Start frame.
        end: int
            End frame.
        codec: str
            Codec to use for saving the video.
            e.g. "MJPG" or "XVID"
        """
        cap = self.video
        save_path = self.vid_path.split(".")[0]

        # validate input
        if start < 0:
            raise ValueError("Start frame must be non-negative.")
        if end < start:
            raise ValueError("End frame must be greater than or equal to start frame.")
        total_frames = int(cap.get(CAP_PROP_FRAME_COUNT))
        if end >= total_frames:
            raise ValueError(
                f"End frame must be less than total frames ({total_frames})."
            )

        # setup the video writer
        w_frame, h_frame = (
            int(cap.get(CAP_PROP_FRAME_WIDTH)),
            int(cap.get(CAP_PROP_FRAME_HEIGHT))
        )
        fps = cap.get(CAP_PROP_FPS)
        cap.set(CAP_PROP_POS_FRAMES, start)
        fourcc = VideoWriter_fourcc(*codec)
        out = VideoWriter(
            join(save_path, "_sliced.avi"), fourcc, fps, (w_frame, h_frame)
        )
        counter = 0
        # read until the 'end'
        try:
            while cap.isOpened() and counter <= end:
                ret, frame = cap.read()
                counter += 1
                if ret:
                    if counter >= start:
                        out.write(frame)
                else:
                    break
        except Exception as error:
            print(f"Error occured: {error}")
        finally:
            cap.release()
            out.release()


class Params4Motion(Enum):
    """Parameters for motion analysis."""

    MT_CUTOFF = (7, 7, 15)  # (A, B, C)
    HEIGHT_WIDTH = (300, 1000)  # (height, width)
    SIGMA = 1  # sigma for Gaussian blur for fz function
    SHOCKTIME = 180  # timing for shock, which is 3 minutes
    FREEZE_THRESH = (3500, 3500, 400)  # (A, B, C) threshold for freezing in pixels
    MIN_DURATION = 20  # minimum duration for freezing in frames
    SHOCK_DELAY = (
        35,
        10,
        10,
    )  # (A, B, C) delay for shock onset in frames given observed delay in applying shock


class Params4Motion(Enum):
    """Parameters for motion analysis."""

    MT_CUTOFF = (7, 7, 15)  # (A, B, C)
    HEIGHT_WIDTH = (300, 1000)  # (height, width)
    SIGMA = 1  # sigma for Gaussian blur for fz function
    SHOCKTIME = 180  # timing for shock, which is 3 minutes
    FREEZE_THRESH = (3500, 3500, 400)  # (A, B, C) threshold for freezing in pixels
    MIN_DURATION = 20  # minimum duration for freezing in frames
    SHOCK_DELAY = (
        35,
        10,
        10,
    )  # (A, B, C) delay for shock onset in frames given observed delay in applying shock
