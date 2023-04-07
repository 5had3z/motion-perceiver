"""Export pytorch predictions to numpy file
to then import to waymo"""

from argparse import ArgumentParser
from pathlib import Path
import os


def main():
    parser = ArgumentParser()
    parser.add_argument("-s", "--workspace", type=Path)
    parser.add_argument("-x", "--run_hash", type=str)
    parser.add_argument("-c", "--config_file", type=Path)
    parser.add_argument("--split", type=str, choices=["test", "val"])

    subparsers = parser.add_subparsers(dest="command")
    infer = subparsers.add_parser("generate", help="Run pytorch inference")
    infer.add_argument("-w", "--workers", type=int, default=4)
    infer.add_argument("--batch_size", type=int, default=16)

    evaluate = subparsers.add_parser(
        "evaluate", help="Run comparison between pytorch and waymo eval"
    )
    evaluate.add_argument("--visualize", action="store_true")

    export = subparsers.add_parser("export", help="Export to waymo evaluation server")

    args = parser.parse_args()

    split = {"test": "testing", "val": "validation"}[args.split]  # Expand string
    gt_path = args.workspace / f"{split}_ground_truth"
    pred_path = args.workspace / args.run_hash / f"{split}_blobs"
    id_path = (
        Path(os.environ.get("DATAPATH", "/data"))
        / f"challenge_{split}_scenario_ids.txt"
    )
    with open(id_path, "r", encoding="utf-8") as f:
        scenario_ids = set([l.strip() for l in f.readlines()])

    if not gt_path.exists():
        gt_path.mkdir()

    if not pred_path.exists():
        pred_path.mkdir()

    if args.command == "generate":
        from konductor.trainer.init import cli_init_config
        from utils.export_torch import initialize, run_export

        exp_cfg = cli_init_config(args)
        model, dataloader = initialize(exp_cfg, args)
        run_export(model, dataloader, scenario_ids, pred_path, gt_path, args)

    elif args.command == "evaluate":
        from utils.export_tf import evaluate_methods

        evaluate_methods(id_path, pred_path, split, args.visualize)

    elif args.command == "export":
        from utils.export_tf import export_evaluation

        export_evaluation(pred_path)


if __name__ == "__main__":
    main()
