from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from torch import Tensor, uint8
from torchvision.transforms.functional import normalize
from torchvision.utils import flow_to_image


def reverse_image_transforms(image: Tensor) -> Tensor:
    """
    Reverse image normalization, permute RGB -> BGR and rescale [0, 1] -> [0, 255].
    This is useful for visualising an image yielded from a dataloader that has the
    aformentioned typical data normalisation transforms.
    Channel permutation is because OpenCV....
    Where original normalisation had the typical parameters:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    """
    image = normalize(
        image,
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
        inplace=False,
    )
    image = (255 * image[:, [2, 1, 0]]).clamp(0, 255).to(uint8)
    return image


def apply_ts_text(ts: float, frame: np.ndarray, extra: str = "") -> np.ndarray:
    """Apply timestamp text to image frame and optional "extra" text underneath"""
    text_args = [cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2, cv2.LINE_AA]

    if ts < 0:
        text = "past"
    elif ts == 0:
        text = "present"
    else:
        text = "future"

    frame = cv2.putText(frame, f"{text}: {ts:+}", (25, 50), *text_args)

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
    timestamps: List[float],
    path: Path,
    thresh: float | None = None,
    roadmap: np.ndarray | None = None,
    signals: Tuple[np.ndarray, np.ndarray] | None = None,
    roadmap_scale: float = 1.0,
) -> None:
    """Write video of prediction over time"""

    video_shape = (800, 800)
    v_writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"VP90"), 10, video_shape
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
                roadmap = np.moveaxis(roadmap, [0], [-1])
            else:
                raise NotImplementedError("Case not handled")

        # Center-Crop the roadmap to the ROI if necessary
        if roadmap_scale < 1.0:
            start = int((1 - roadmap_scale) * roadmap.shape[0] / 2)
            end = start + int(roadmap_scale * roadmap.shape[0])
            roadmap = roadmap[start:end, start:end]

        # Resize the roadmap to the output shape
        roadmap = cv2.resize(roadmap, video_shape, interpolation=cv2.INTER_NEAREST)

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
            timestamps[idx],
            bgr_frame,
            extra=f"pr>{thresh}" if thresh else "",
        )
        v_writer.write(bgr_frame)

    v_writer.release()


def create_flow_frame(
    pred_flow: Tensor,
    pred_occ: Tensor,
    truth_flow: Tensor,
    frame_size: Tuple[int, int],
    mask_thresh: float = 0.5,
    roadmap: Tensor | None = None,
) -> np.ndarray:
    """Create a side-by-side frame of predicted occupancy flow and ground truth,
    mask out predicted flow with predicted occupancy over a threshold
    A threshold of zero is obviously no threshold (show all flow for every pixel)
    """
    pred_flow_rgb = flow_to_image(pred_flow)
    pred_flow_rgb[:, pred_occ < mask_thresh] = 255  # set to white
    truth_flow_rgb = flow_to_image(truth_flow)

    if roadmap is not None:
        if roadmap.ndim == 2:
            apply_roadmap_to_frame(pred_flow_rgb, roadmap, mask_thresh)
            apply_roadmap_to_frame(truth_flow_rgb, roadmap, mask_thresh)
        else:
            raise NotImplementedError()

    rgb_frame = cv2.hconcat(
        [
            np.moveaxis(pred_flow_rgb.cpu().numpy(), 0, 2),
            np.moveaxis(truth_flow_rgb.cpu().numpy(), 0, 2),
        ]
    )
    rgb_frame = cv2.resize(rgb_frame, frame_size, interpolation=cv2.INTER_LINEAR)
    return rgb_frame


def write_flow_video(
    pred_flow_sequence: Tensor,
    pred_occ_sequence: Tensor,
    truth_flow_sequence: Tensor,
    timestamps: List[float],
    path: Path,
    mask_thresh: float = 0.5,
    roadmap: Tensor | None = None,
):
    """"""
    video_shape = (1600, 800)
    v_writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"VP90"), 10, video_shape
    )

    if not v_writer.isOpened():
        raise RuntimeError(f"Can't write video, writer not open: {path}")

    for idx in range(pred_flow_sequence.shape[1]):
        rgb_frame = create_flow_frame(
            pred_flow_sequence[:, idx],
            pred_occ_sequence[idx],
            truth_flow_sequence[:, idx],
            video_shape,
            mask_thresh,
            roadmap,
        )

        rgb_frame = apply_ts_text(timestamps[idx], rgb_frame, extra=f"pr>{mask_thresh}")

        v_writer.write(rgb_frame)

    # Delete torch tensors https://pytorch.org/docs/stable/multiprocessing.html
    del pred_flow_sequence, pred_occ_sequence, truth_flow_sequence
    import gc

    # Make sure they're gone
    gc.collect()
    torch.cuda.empty_cache()

    v_writer.release()
