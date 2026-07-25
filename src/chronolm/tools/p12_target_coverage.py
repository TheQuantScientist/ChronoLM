"""Print target-coverage statistics for the APN P12 test split."""

from __future__ import annotations

import numpy as np

from chronolm.apn import add_apn_to_path

add_apn_to_path()
from data.dependencies.tsdm.tasks.P12 import Physionet2012  # noqa: E402


def main() -> None:
    print("Loading PhysioNet P12 data...")
    task = Physionet2012(seq_len=36, pred_len=3)
    columns = list(task.dataset.columns)
    test_dataset = task.get_dataset((0, "test"))
    n_samples = len(test_dataset)

    print(f"Test samples: {n_samples}\n")
    print(
        f"{'Variable':<15} {'History_mean':>12} "
        f"{'Target_mean':>12} {'Has_target':>12}"
    )
    print("=" * 55)

    for variable_index, variable_name in enumerate(columns):
        history_counts = []
        target_counts = []

        for sample_index in range(n_samples):
            sample = test_dataset[sample_index]
            _, x_input, _ = sample.inputs
            y_target = sample.targets
            history_counts.append((~x_input[:, variable_index].isnan()).sum().item())
            target_counts.append((~y_target[:, variable_index].isnan()).sum().item())

        samples_with_target = sum(1 for count in target_counts if count > 0)
        print(
            f"{variable_name:<15} {np.mean(history_counts):>12.1f} "
            f"{np.mean(target_counts):>12.2f} {samples_with_target:>8}/{n_samples}"
        )


if __name__ == "__main__":
    main()
