from typing import Tuple
from pathlib import Path

import cv2
import numpy as np


def apply_ts_text(ts: int, frame: np.ndarray, extra: str = "") -> np.ndarray:
    """Apply timestamp text to image frame and optional "extra" text underneath"""
    text_args = [cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2, cv2.LINE_AA]

    if ts < 10:
        text = "past"
    elif ts == 10:
        text = "present"
    else:
        text = "future"
    text += f": {ts-10:+}"

    frame = cv2.putText(frame, text, (25, 50), *text_args)

    if extra:
        frame = cv2.putText(frame, extra, (25, 75), *text_args)

    return frame


def create_occupancy_frame(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    resolution: Tuple[int, int],
    threshold: float | None = None,
) -> np.ndarray:
    """Create an rgb frame showing the ground truth and predicted frames"""
    bgr_frame = 255 * np.ones((*resolution, 3), dtype=np.uint8)
    prediction = cv2.resize(prediction, resolution, interpolation=cv2.INTER_LINEAR)
    ground_truth = cv2.resize(ground_truth, resolution, interpolation=cv2.INTER_NEAREST)

    # Red for false negatives
    bgr_frame[ground_truth == 1] = np.array((0, 0, 200), dtype=np.uint8)

    # Blue for false positives
    if threshold is not None:
        bgr_frame[prediction > threshold] = np.array((255, 0, 0), dtype=np.uint8)
    else:
        mask = ground_truth == 0
        rg = (255 * prediction).astype(np.uint8)[mask]
        b = np.zeros_like(rg)
        # subtract rg from prediction
        bgr_frame[mask] -= np.stack([b, rg, rg], axis=-1)

    # Green for true positives
    if threshold is not None:
        bgr_frame[(prediction > threshold) & (ground_truth == 1)] = np.array(
            (0, 255, 0), dtype=np.uint8
        )
    else:
        mask = (prediction > 0.5) & (ground_truth == 1)
        rb = ((1 - prediction) * 255).astype(np.uint8)[mask]
        g = 200 * np.ones_like(rb)
        bgr_frame[mask] = np.stack([rb, g, rb], axis=-1)

    return bgr_frame


def apply_roadmap_to_frame(
    frame: np.ndarray, roadmap: np.ndarray, threshold: float | None = None
):
    """Overlay roadmap on rgb image"""
    mask = roadmap > 0
    road_thresh = 255 * threshold if threshold is not None else 255 / 2
    for ch in range(3):
        mask &= frame[..., ch] > road_thresh  # mostly white
    frame[mask] = np.array((0, 0, 0), dtype=np.uint8)


def apply_image_to_frame(
    frame: np.ndarray, image: np.ndarray, threshold: float | None = None
):
    """Apply rgb context image to occupancy frame"""
    mask = np.ones(image.shape[:-1], dtype=bool)
    road_thresh = 255 * threshold if threshold is not None else 255 / 2
    for ch in range(3):
        mask &= frame[..., ch] > road_thresh  # mostly white
    frame[mask] = image[mask]


def signal_color(cls: float):
    match int(cls):
        case 1 | 4:  # arrow stop | stop
            return (0, 0, 255)
        case 2 | 5:  # arrow caution | caution
            return (0, 255, 255)
        case 3 | 6:  # arrow go | go
            return (0, 255, 0)
        case _:  # grey for other
            return (128, 128, 128)


def apply_signals_to_frame(frame: np.ndarray, signals: np.ndarray, scale: float):
    """Overlay signals on rgb image"""
    hw = frame.shape[0]
    for signal in signals:
        pos = tuple(int(x) for x in ((signal[:2] / scale + 1) * hw / 2))
        frame = cv2.circle(frame, pos, 6, signal_color(signal[-1]), -1)
    return frame


def write_occupancy_video(
    data: np.ndarray,
    pred: np.ndarray,
    path: Path,
    thresh: float | None = None,
    roadmap: np.ndarray | None = None,
    signals: Tuple[np.ndarray, np.ndarray] | None = None,
    roadmap_scale: float = 1.0,
    time_stride: int = 1,
) -> None:
    """Write video of prediction over time"""

    video_shape = (800, 800)
    v_writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"VP90"), 10 // time_stride, video_shape
    )

    if not v_writer.isOpened():
        raise RuntimeError(f"Can't write video, writer not open: {path}")

    if roadmap is not None:
        if roadmap.ndim == 3:
            # Squeeze channel if its roadgraph raster
            if roadmap.shape[0] == 1:
                roadmap = roadmap.squeeze(0)
            # Reshape to channels last otherwise
            elif roadmap.shape[0] == 3:
                roadmap = np.transpose(roadmap, [2, 1, 0])
            else:
                raise NotImplementedError("Case not handled")

        # Center-Crop the roadmap to the ROI if necessary
        if roadmap_scale < 1.0:
            start = int((1 - roadmap_scale) * roadmap.shape[0] / 2)
            end = start + int(roadmap_scale * roadmap.shape[0])
            roadmap = roadmap[start:end, start:end]

        # Resize the roadmap to the output shape
        roadmap = cv2.resize(roadmap, video_shape, interpolation=cv2.INTER_NEAREST)

    # Check if prediction is 2phase, if so squeeze ground truth
    if pred.shape[0] == 19:
        data[11:19] = data[20::10]
        data = data[:19]

    for idx, (pred_frame, data_frame) in enumerate(zip(pred, data)):
        bgr_frame = create_occupancy_frame(data_frame, pred_frame, video_shape, thresh)
        if roadmap is not None:
            if roadmap.ndim == 2:
                apply_roadmap_to_frame(bgr_frame, roadmap, thresh)
            else:
                apply_image_to_frame(bgr_frame, roadmap, thresh)

        if signals is not None:
            masked_signals = signals[0][idx][signals[1][idx]]
            bgr_frame = apply_signals_to_frame(bgr_frame, masked_signals, roadmap_scale)

        bgr_frame = apply_ts_text(
            idx * time_stride, bgr_frame, extra=f"pr>{thresh}" if thresh else ""
        )
        v_writer.write(bgr_frame)

    v_writer.release()
