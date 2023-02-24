# Motion Perceiver

## Run Training

You can set environment variables permanently or add them on execution.
```bash
DATAPATH=/path/to/dataset PRETRAINED_ROOT=/path/to/pretrained python3 train.py --config_file cfg/waymo_sdc_frame.yml --workspace ./checkpoints --workers 4 --pbar
```
