"""Inspect the APN P12 split objects."""

from __future__ import annotations

from chronolm.apn import add_apn_to_path

add_apn_to_path()
from data.dependencies.tsdm.tasks.P12 import Physionet2012  # noqa: E402


def describe_part(index: int, part: object) -> None:
    print(f"  Part [{index}]: type={type(part)}")
    if hasattr(part, "shape"):
        print(f"    shape={part.shape}, dtype={part.dtype}")
        print(f"    min={part.min():.4f}, max={part.max():.4f}, mean={part.mean():.4f}")
    elif hasattr(part, "__len__"):
        print(f"    len={len(part)}")
        if hasattr(part, "columns"):
            print(f"    columns={list(part.columns)[:10]}")
            print(f"    head:\n{part.head()}")
        elif hasattr(part, "keys"):
            keys = list(part.keys())[:5]
            print(f"    keys (first 5): {keys}")
            for key in keys[:2]:
                value = part[key]
                shape = value.shape if hasattr(value, "shape") else len(value)
                print(f"    [{key}]: type={type(value)}, shape={shape}")


def main() -> None:
    task = Physionet2012()
    train, _, test = task.splits

    print("=== TRAIN ===")
    for index, part in enumerate(train):
        describe_part(index, part)

    print("\n=== TEST ===")
    for index, part in enumerate(test):
        describe_part(index, part)


if __name__ == "__main__":
    main()
