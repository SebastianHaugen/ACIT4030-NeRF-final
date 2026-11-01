"""
Short training run that checks the whole pipeline works before a full run.
Results go to outputs/smoke_test, so the real results are never overwritten.
"""
import statistics
import sys

from src.train import main as train
from src.repo_util.LoadConfigurations import (
    ACTIVE_DATASET,
    EXPERIMENT,
    SMOKE_N_ITERATIONS,
    SMOKE_VISUALIZE_EVERY,
    SMOKE_OUTPUT_DIR,
    SMOKE_MIN_LOSS_DROP,
)

def main():
    """Trains for a few hundred iterations and checks that the loss goes down."""
    output_dir = SMOKE_OUTPUT_DIR / EXPERIMENT
    print(f"Smoke test on {ACTIVE_DATASET} ({EXPERIMENT}): {SMOKE_N_ITERATIONS} iterations -> {output_dir}")
    history = train(
        n_iter=SMOKE_N_ITERATIONS,
        output_dir=output_dir,
        vis_every=SMOKE_VISUALIZE_EVERY,
        checkpoint_every=SMOKE_N_ITERATIONS,
    )

    # Compares the average loss over the first and the last 10 % of the run
    total = history["total"]
    window = max(10, len(total) // 10)
    start = statistics.mean(total[:window])
    end = statistics.mean(total[-window:])
    drop = 1.0 - end / start

    print(f"\nLoss: {start:.4f} (start) -> {end:.4f} (end), {drop:.0%} lower")
    # Stops with an error code if the loss did not drop enough
    if drop < SMOKE_MIN_LOSS_DROP:
        print(f"FAIL: expected at least {SMOKE_MIN_LOSS_DROP:.0%} lower. "
              f"Check the previews in {SMOKE_OUTPUT_DIR / 'previews'}.")
        sys.exit(1)
    print("PASS: the model is learning. Safe to start the full run.")


if __name__ == "__main__":
    main()