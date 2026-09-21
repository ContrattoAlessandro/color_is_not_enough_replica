import argparse
from .utils import read_config


def main():
    import torch
    torch.set_num_threads(4)
    parser = argparse.ArgumentParser(description="Color Is Not Enough: selected YOLOv8s joint model")
    parser.add_argument("--config", help="Training config; evaluation/resume use the checkpoint config by default")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    t = sub.add_parser("train")
    t.add_argument("--resume")
    t.add_argument("--epochs", type=int)
    t.add_argument("--limit", type=int)
    t.add_argument("--output")
    t.add_argument("--batch-size", type=int)
    t.add_argument("--evaluate-after", action="store_true")
    e = sub.add_parser("evaluate")
    e.add_argument("--checkpoint", required=True)
    e.add_argument("--output")
    e.add_argument("--limit", type=int)
    a = sub.add_parser("audit", help="Evaluate a checkpoint on validation and report sequence/scene errors")
    a.add_argument("--checkpoint", required=True)
    a.add_argument("--output", default="artifacts/validation_audit")
    p = sub.add_parser("predict")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--output", default="artifacts/prediction")
    args = parser.parse_args()
    from pathlib import Path
    checkpoint_path = getattr(args, "checkpoint", None) or getattr(args, "resume", None)
    if args.config:
        config = read_config(args.config)
    elif checkpoint_path:
        config = torch.load(checkpoint_path, map_location="cpu", weights_only=False)["config"]
    else:
        config = read_config("configs/dtld.yaml")
    if args.command == "prepare":
        from .data import prepare
        if config.get('validation', {}).get('enabled'):
            raise ValueError('Use python -m cine.split to create the clean split; do not overwrite it with the original full-training preparation.')
        prepare(config)
    elif args.command == "train":
        from .engine import train
        from pathlib import Path
        from .utils import write_json
        if args.resume and not args.output:
            args.output = str(Path(args.resume).resolve().parent)
        output_dir = Path(args.output or config["train"]["output"])
        try:
            checkpoint = train(config, args.resume, args.epochs, args.limit, args.output, args.batch_size)
            if args.evaluate_after:
                from .evaluate import evaluate
                selected = output_dir / "best.pt"
                if config.get("validation", {}).get("enabled") and selected.exists():
                    checkpoint = selected
                write_json(output_dir / "status.json", dict(phase="evaluating_selected_checkpoint", checkpoint=str(checkpoint)))
                metrics = evaluate(config, checkpoint)
                write_json(output_dir / "status.json", dict(phase="complete", checkpoint=str(checkpoint), metrics=str(output_dir / "evaluation" / "metrics.json")))
        except Exception as error:
            write_json(output_dir / "failure.json", dict(error=type(error).__name__, message=str(error)))
            raise
    elif args.command == "evaluate":
        from .evaluate import evaluate
        evaluate(config, args.checkpoint, args.output, args.limit)
    elif args.command == "audit":
        from .generalization import audit_checkpoint
        audit_checkpoint(args.checkpoint, args.output)
    else:
        from .evaluate import predict
        predict(args.checkpoint, args.image, args.output)


if __name__ == "__main__":
    main()
