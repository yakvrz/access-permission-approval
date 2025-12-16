import argparse

from prefect import flow

from .monitoring import run_monitoring
from .train import run_training


@flow(name="access-permission-training")
def training_flow(config: str = "config/default.yaml", monitor: bool = True):
    result = run_training(config)
    if monitor:
        run_monitoring(config)
    return result.metrics


@flow(name="access-permission-monitoring")
def monitoring_flow(config: str = "config/default.yaml", current_interactions: str | None = None):
    return run_monitoring(config, current_interactions=current_interactions)


def main():
    parser = argparse.ArgumentParser(description="Prefect-managed pipelines for access permission ML.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Run training flow")
    train_parser.add_argument("--config", default="config/default.yaml")
    train_parser.add_argument("--skip-monitor", action="store_true", help="Skip monitoring after training")

    monitor_parser = subparsers.add_parser("monitor", help="Run monitoring flow")
    monitor_parser.add_argument("--config", default="config/default.yaml")
    monitor_parser.add_argument(
        "--current-interactions",
        default=None,
        help="Override interactions CSV for monitoring window",
    )

    args = parser.parse_args()
    if args.command == "train":
        training_flow(config=args.config, monitor=not args.skip_monitor)
    elif args.command == "monitor":
        monitoring_flow(config=args.config, current_interactions=args.current_interactions)


if __name__ == "__main__":
    main()
