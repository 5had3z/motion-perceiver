from pathlib import Path
from typing import Dict

import cv2
import numpy as np
import torch
from torch import Tensor
from matplotlib import pyplot as plt
from torchvision.utils import flow_to_image


def occupancy_sequence(data: Dict[str, Tensor]):
    """Print images of occpancy for each class over time"""
    for i, (heatmap, bt_idx) in enumerate(zip(data["heatmap"], data["time_idx"])):
        occ: np.ndarray = heatmap.cpu().to(torch.uint8).numpy() * 255
        save_dir = Path(f"occupancy_{i}")
        if not save_dir.exists():
            save_dir.mkdir()
        for occ_c, cls_ in zip(occ, ["vehicle", "pedestrian", "cyclist"]):
            for occ_t, t_idx in zip(occ_c, bt_idx):
                cv2.imwrite(
                    str(save_dir / f"{cls_}_occupancy_{t_idx}.png"),
                    cv2.flip(occ_t, 0),
                )


def scatterplot_sequence(data: Dict[str, Tensor], cur_idx: int = 10) -> None:
    """Plot world-lines of instances"""
    for i, (seq, valid) in enumerate(zip(data["agents"], data["agents_valid"])):
        plt.figure(f"agents_{i}", figsize=(20, 20))
        plt.xlim((-1, 1))
        plt.ylim((-1, 1))
        for inst, mask in zip(seq, valid):
            valid_pos = inst[mask != 0].cpu().numpy()
            plt.scatter(valid_pos[..., 0], valid_pos[..., 1])
            if inst.shape[0] > 1 and mask[cur_idx] != 0:
                xytvxvy = inst[cur_idx, 0:5].cpu().numpy()

                # Check velocity
                dx = 1 / 40 * xytvxvy[3]
                dy = 1 / 40 * xytvxvy[4]
                plt.arrow(xytvxvy[0], xytvxvy[1], dx, dy, width=0.005, color="b")

                # Check Headding
                dx = 0.02 * np.cos(np.pi * xytvxvy[2])
                dy = 0.02 * np.sin(np.pi * xytvxvy[2])
                plt.arrow(xytvxvy[0], xytvxvy[1], dx, dy, width=0.005, color="r")

        signals = data["signals"][i, ..., :2]
        mask = data["signals_valid"][i]
        signals = signals[mask.bool()].cpu().numpy()
        plt.scatter(signals[:, 0], signals[:, 1], marker="x", s=500)

        plt.tight_layout()
        plt.savefig(f"agents_{i}.png")
        plt.close(f"agents_{i}")


def pose_to_poly(pose_data: np.ndarray) -> np.ndarray:
    """
    Generates polypoints for polyfill to create occupancy mask
    Pose data consists of [val,x,y,t,dx,dy,dt,l,w]
    return four points that describe the mask polygon
    """
    # from pose make box [tl, tr, br, bl]
    bbox = np.array(
        [
            [-pose_data[-1], +pose_data[-2]],  # -w, +l
            [+pose_data[-1], +pose_data[-2]],  # +w, +l
            [+pose_data[-1], -pose_data[-2]],  # +w, -l
            [-pose_data[-1], -pose_data[-2]],  # -w, -l
        ]
    ).transpose()
    rot_mat = np.array(
        [
            [np.cos(pose_data[3]), -np.sin(pose_data[3])],
            [np.sin(pose_data[3]), np.cos(pose_data[3])],
        ]
    )
    return rot_mat @ bbox + pose_data[1:3, None]


def render_occupancy(
    sample_data: Tensor, scale: float = 1.0, padding: float = 40.0
) -> np.ndarray:
    """
    Generate occupancy heatmap from data [inst, 1, pose]
    scale: param increases resolution
    padding: pad the occupancy image
    """
    # generate occupancy map with resolution
    min_x, max_x = (
        torch.min(sample_data[..., 1]).item(),
        torch.max(sample_data[..., 1]).item(),
    )
    min_y, max_y = (
        torch.min(sample_data[..., 2]).item(),
        torch.max(sample_data[..., 2]).item(),
    )

    x_size = max_x - min_x + padding
    y_size = max_y - min_y + padding

    occupancy_map = np.zeros(
        (int(y_size * scale), int(x_size * scale), 1), dtype=np.uint8
    )
    translate_ = np.array([[min_x - padding / 2], [min_y - padding / 2]])
    sample_np = sample_data.cpu().numpy()
    for sample in sample_np:
        poly_pts: np.ndarray = (pose_to_poly(sample) - translate_) * scale
        poly_pts = poly_pts.transpose().astype(np.int64)
        cv2.fillPoly(occupancy_map, pts=[poly_pts], color=255)

    return occupancy_map


def occupancy_from_current_pose(data: Dict[str, Tensor]) -> None:
    """Requires loading of raw waymo tf data i.e. raw "current" frame"""
    for i, state_ in enumerate(data["current"]):
        valid_mask = state_[..., 0] != 0
        valid_data = state_[valid_mask]
        occ = render_occupancy(valid_data)
        plt.figure(f"occupancy_{i}", figsize=(20, 20))
        # invert y so min-y is bottom of the image to match scatter plot
        occ = cv2.flip(occ, 0)
        plt.imshow(occ)
        plt.tight_layout()
        plt.savefig(f"occupancy_{i}.png")


def roadgraph(roadgraph: Tensor, valid_mask: Tensor) -> None:
    """Plot roadgraph segments from raw data"""

    for idx, (rg, mask) in enumerate(zip(roadgraph, valid_mask)):
        rg_filt = rg[mask.bool()[:, 0]]
        ids = rg_filt[:, -1].unique()

        figname = f"lanecenter_{idx}"
        plt.figure(figname, figsize=(20, 20))
        for id in ids:
            id_feats = rg_filt[rg_filt[:, -1] == id]
            if id_feats[0, -2].item() in {1, 2, 3}:
                plt.plot(id_feats[:, 0].cpu(), id_feats[:, 1].cpu())
        plt.tight_layout()
        plt.savefig(f"{figname}.png")
        plt.close(figname)

        figname = f"roadline_{idx}"
        plt.figure(figname, figsize=(20, 20))
        for id in ids:
            id_feats = rg_filt[rg_filt[:, -1] == id]
            if id_feats[0, -2].item() in set(range(6, 14)):
                plt.plot(id_feats[:, 0].cpu(), id_feats[:, 1].cpu())
        plt.tight_layout()
        plt.savefig(f"{figname}.png")
        plt.close(figname)


def roadmap(roadmap: Tensor) -> None:
    """Visualise the roadmap image"""
    for idx, im in enumerate(roadmap):
        figname = f"lanecenter_image_{idx}"
        plt.figure(figname, figsize=(20, 20))
        plt.imshow(im.cpu()[0])
        plt.tight_layout()
        plt.savefig(f"{figname}.png")
        plt.close(figname)


def render_signals(signals: np.ndarray, im_shape) -> np.ndarray:
    """"""
    signal_image = np.zeros(im_shape)
    for signal in signals:
        pos = tuple(int(x) for x in ((signal + 1) * im_shape / 2))
        signal_image = cv2.drawMarker(
            signal_image, pos, 1, cv2.MARKER_TILTED_CROSS, markerSize=5
        )
    return signal_image


def resize_occupancy(occ: np.ndarray, roi_scale: float) -> np.ndarray:
    """Resize occupancy to map scale if there is decoupled scales"""
    if roi_scale == 1:
        return occ
    scaled = np.zeros_like(occ)
    start = int((1 - roi_scale) * occ.shape[0] / 2)
    end = start + int(roi_scale * occ.shape[0])
    scaled[start:end, start:end] = cv2.resize(
        occ, tuple(s // 2 for s in occ.shape), interpolation=cv2.INTER_LINEAR
    )
    return scaled


def optical_flow(flow: Tensor) -> None:
    """Writes images of the flow output as an HSV Image
    Input is assumed as N,T,2,H,W"""
    for tidx in range(flow.shape[2]):
        flow_img = flow_to_image(flow[:, :, tidx])
        for bidx in range(flow.shape[0]):
            cv2.imwrite(
                f"flow_{bidx}_{tidx}.png",
                np.moveaxis(flow_img[bidx].cpu().numpy(), 0, 2),
            )


def roadmap_and_occupancy(
    roadmaps: Tensor, occupancies: Tensor, signals: Tensor, roi_scale: float = 1
) -> None:
    """Overlay both occupancy and roadmap image to ensure they're synchronised"""
    roadmaps = roadmaps.cpu().numpy()
    occupancies = occupancies.cpu().numpy()
    signals = signals.cpu().numpy()[:, :, 0, 0:2]  # xy same for all time

    for bidx, (roadmap, occupancy_vec, signal) in enumerate(
        zip(roadmaps, occupancies, signals)
    ):
        signal_map = render_signals(signal, occupancy_vec.shape[-2:])
        roadmap = cv2.resize(
            roadmap[0], occupancy_vec.shape[-2:], interpolation=cv2.INTER_LINEAR
        )
        for tidx, occupancy in enumerate(occupancy_vec[0]):
            figname = f"occupancy_roadimg_{bidx}_{tidx}"
            plt.figure(figname, figsize=(10, 10))
            img = np.stack(
                [roadmap, resize_occupancy(occupancy, roi_scale), signal_map], axis=-1
            )
            if tidx == 1:  # SDC Target Frame
                img = cv2.drawMarker(
                    img,
                    (img.shape[0] // 2, img.shape[1] // 2 + 64),
                    (1, 1, 1),
                    cv2.MARKER_TILTED_CROSS,
                    markerSize=6,
                    thickness=2,
                )
            plt.imshow(img)
            plt.tight_layout()
            plt.savefig(f"{figname}.png")
            plt.close(figname)
