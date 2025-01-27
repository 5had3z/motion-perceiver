# Motion Perceiver: Real-Time Occupancy Forecasting for Embedded Systems 

### 🥇 1st - Observed Occupancy Soft-IoU Waymo Open Motion (22/08/23)
### 🥉 3rd - Observed Occupancy AUC Waymo Open Motion (26/12/23)

[waymo leaderboard](https://waymo.com/open/challenges/2022/occupancy-flow-prediction-challenge/)

## Abstract - [arXiv](https://arxiv.org/abs/2306.08879)

This work introduces a novel and adaptable architecture designed for real-time occupancy foXecasting that outperforms existing state-of-the-art models on the Waymo Open Motion Dataset in Soft IOU. The proposed model uses recursive latent state estimation with learned transformer-based functions to effectively update and evolve the state. This enables highly efficient real-time inference on embedded systems, as profiled on an Nvidia Xavier AGX. Our model, MotionPerceiver, achieves this by encoding a scene into a latent state that evolves in time through self-attention mechanisms. Additionally, it incorporates relevant scene observations, such as traffic signals, road topology and agent detections, through cross-attention mechanisms. This forms an efficient data-streaming architecture, that contrasts with the expensive, fixed-sequence input common in existing models. The architecture also offers the distinct advantage of generating occupancy predictions through localized querying based on a position of interest, as opposed to generating fixed-size occupancy images, including potentially irrelevant regions.

### Citation

```bibtext
@ARTICLE{10417132,
  author={Ferenczi, Bryce and Burke, Michael and Drummond, Tom},
  journal={IEEE Robotics and Automation Letters}, 
  title={MotionPerceiver: Real-Time Occupancy Forecasting for Embedded Systems}, 
  year={2024},
  volume={9},
  number={3},
  pages={2822-2829},
  keywords={Forecasting;Predictive models;Trajectory;Real-time systems;Encoding;Tracking;Prediction algorithms;Computer vision for transportation;deep learning for visual perception;representation learning},
  doi={10.1109/LRA.2024.3360811}}
```


## Model and Inference Diagram

![ArchImage](./media/arch.svg "MotionPerceiver Architecture")


## Inference Sample - [More Here](https://sites.google.com/monash.edu/motionperceiver)

![TwoPhase](./media/sample.gif "TwoPhase")

## General Notes

Most relevant commands should have reasonable docstrings attached, I use typer so just use `python3 script.py command --help` to figure out what a script command does and its arguments (or look at the code).

If you have problems running this bare-metal (you'll need gcc-13), you should either use the main dockerfile included, or use it as instructions on how to install dependencies.

## Training

You just need to download the waymo training shards. You can set environment variables permanently or add them on execution.
```bash
DATAPATH=/path/to/dataset PRETRAINED_ROOT=/path/to/pretrained python3 train.py --config_file cfg/waymo_two_phase.yml --workspace ./checkpoints --workers 4 --epoch 75 --pbar
```

### Dataset Plugins

You will need OpenCV, TBB and gcc-13. To build the dataloader plugins, you can either invoke cmake manually, or use the VSCode CMake Extension. When using the VSCode Extension, you can specify a different python executable in `settings.json`, this is helpful if your main environment is in a venv or something.

```json
{
    "cmake.configureArgs": [
        "-DPython3_EXECUTABLE=${userHome}/.venv/bin/python"
    ]
}
```

## Testing + Tools

### Make Videos

To create the videos, simply run `tools.py make-video`.

### Make Submission or Evaluate on Validation

Since I prefer working in PyTorch land, that is what the the majority of the codebase is written in. However since Waymo is an Alphabet company, they dogfood their own products and use TensorFlow. To separate TensorFlow and PyTorch runtime usage, evaluation is done in two steps. First use the `evaluate.py generate` command which saves the output as numpy files, `evaluate.py waypoint-evaluate` then reads these numpy files and calculates and logs the metrics .`evaluate.py export` creates a waymo submission. I've included a docker compose file to run tensorflow things in a container.

## Footnotes

There is some WIP code on pedestrain datasets.

The `occupancy flow` ground truth generated by this code isn't the same flow as the waymo challenge, it is a simplified verison which does not consider vehicle heading. Hence, it is a few pixels off for vehicles that are turning. The large absolute error values are the parts where the raster mask itself differs, the actual difference due to rotation is <5px end-point-error. Run test_waymo_native.py to inspect the difference for youself.

![FlowDiff](./media/flow_diff.png)
