"""Print observation-count statistics for the APN P12 test split."""

from __future__ import annotations

import numpy as np
import pandas as pd

from chronolm.apn import add_apn_to_path

add_apn_to_path()
from data.dependencies.tsdm.tasks.P12 import Physionet2012  # noqa: E402


def main() -> None:
    print("Loading PhysioNet P12 data...")
    task = Physionet2012(seq_len=36, pred_len=3)
    columns = list(task.dataset.columns)
    test_dataset = task.get_dataset((0, "test"))
    n_samples = len(test_dataset)

    print(f"Test samples: {n_samples}")
    print(f"Variables: {len(columns)}\n")

    variable_observations = {variable: [] for variable in columns}
    total_observations_per_patient = []

    for sample_index in range(n_samples):
        sample = test_dataset[sample_index]
        _, x_input, _ = sample.inputs

        patient_total = 0
        for variable_index, variable_name in enumerate(columns):
            history = x_input[:, variable_index]
            n_valid = (~history.isnan()).sum().item()
            variable_observations[variable_name].append(n_valid)
            patient_total += n_valid

        total_observations_per_patient.append(patient_total)

    print("=" * 70)
    print(
        f"{'Variable':<15} {'Min':>6} {'Mean':>8} "
        f"{'Median':>8} {'Max':>6} {'Has data':>10}"
    )
    print("=" * 70)

    rows = []
    for variable_name in columns:
        observations = variable_observations[variable_name]
        patients_with_data = sum(1 for value in observations if value > 0)
        rows.append(
            {
                "Variable": variable_name,
                "Min": min(observations),
                "Mean": round(float(np.mean(observations)), 1),
                "Median": int(np.median(observations)),
                "Max": max(observations),
                "Patients_with_data": f"{patients_with_data}/{n_samples}",
            }
        )
        print(
            f"{variable_name:<15} {min(observations):>6} "
            f"{np.mean(observations):>8.1f} {int(np.median(observations)):>8} "
            f"{max(observations):>6} {patients_with_data:>5}/{n_samples}"
        )

    print(f"\n{'=' * 70}")
    print("Total observations per patient across all variables:")
    print(f"  Min:    {min(total_observations_per_patient)}")
    print(f"  Mean:   {np.mean(total_observations_per_patient):.1f}")
    print(f"  Median: {int(np.median(total_observations_per_patient))}")
    print(f"  Max:    {max(total_observations_per_patient)}")
    print("=" * 70)

    output_path = "P12_obs_count_36h.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
