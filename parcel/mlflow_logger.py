"""
PARCEL MLflow Logger
====================
Thin wrapper around MLflow so the training script stays clean.
All MLflow imports are local — if mlflow is not installed the logger
degrades gracefully to stdout-only mode.

Usage in train_parcel.py:
    logger = PARCELLogger(experiment_name="parcel")
    logger.start_run(params, args)
    ...
    logger.log(epoch, total_episodes, metrics)   # called each log_interval
    ...
    logger.end_run()
"""

import os


class PARCELLogger:
    """
    Wraps MLflow for experiment tracking.
    Falls back to no-op if mlflow is not installed.
    """

    def __init__(self, experiment_name="parcel", tracking_uri="mlruns"):
        self._active = False
        self._experiment_name = experiment_name
        self._tracking_uri = tracking_uri
        try:
            import mlflow
            self._mlflow = mlflow
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)
            self._available = True
        except ImportError:
            print("[Logger] mlflow not installed — run `pip install mlflow` to enable.")
            print("[Logger] Continuing without MLflow tracking.")
            self._available = False

    def start_run(self, args, extra_tags=None):
        """
        Start an MLflow run and log all hyperparameters.

        args: the argparse Namespace from train_parcel.py
        extra_tags: dict of string tags (e.g. {"status": "debug"})
        """
        if not self._available:
            return

        tags = {"critic_type": args.critic_type, "seed": str(args.seed)}
        if extra_tags:
            tags.update(extra_tags)

        self._mlflow.start_run(tags=tags)
        self._active = True

        # --- Hyperparameters ---
        params = {
            # Critic
            "critic_type":      args.critic_type,
            "gnn_layers":       getattr(args, "gnn_layers", None),
            "edge_dim":         getattr(args, "edge_dim", None),
            "embed_dim":        args.embed_dim,
            "nr_heads":         args.nr_heads,
            "hidden_dim":       args.hidden_dim,

            # Training
            "nr_agents":        args.nr_agents,
            "epochs":           args.epochs,
            "episodes_per_epoch": args.episodes_per_epoch,
            "time_limit":       args.time_limit,
            "lr":               args.lr,
            "clip_ratio":       args.clip_ratio,
            "update_iterations": args.update_iterations,
            "crop_size":        args.crop_size,
            "seed":             args.seed,

            # Curriculum
            "improvement_threshold": args.improvement_threshold,
            "deviation_factor":      args.deviation_factor,
            "sliding_window":        args.sliding_window,

            # Maps — use resolved train/test paths if available
            "train_maps": _map_names(
                getattr(args, "train_maps", None)
                or [args.random_map, args.warehouse_map]
            ),
            "test_maps": _map_names(
                getattr(args, "test_maps", None)
                or getattr(args, "train_maps", None)
                or [args.random_map, args.warehouse_map]
            ),
        }
        self._mlflow.log_params(params)
        print(f"[Logger] MLflow run started  "
              f"(experiment={self._experiment_name}, "
              f"critic={args.critic_type}, seed={args.seed})")

    def log(self, epoch, total_episodes, metrics):
        """
        Log per-epoch metrics.

        metrics: dict with keys from {train_cr, test_cr, test_sr, ralloc}
        """
        if not self._active:
            return
        payload = dict(metrics)
        payload["total_episodes"] = total_episodes
        self._mlflow.log_metrics(payload, step=total_episodes)

    def end_run(self, output_dir=None):
        """Finalize the MLflow run. Optionally log output artifacts."""
        if not self._active:
            return
        if output_dir and os.path.isdir(output_dir):
            self._mlflow.log_artifacts(output_dir, artifact_path="model")
        self._mlflow.end_run()
        self._active = False
        print("[Logger] MLflow run ended.")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.end_run()


def _map_names(paths):
    return "+".join(os.path.basename(p).replace(".map", "") for p in paths if p)
