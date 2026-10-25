
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = ROOT / "scripts"

ORGANIZATION = {
    "data_audit": [
        "audit_dataco.py",
        "inspect_b2b_dates.py",
        "inspect_b2b_model_history.py",
        "inspect_b2b_years.py",
        "inspect_dataco.py",
    ],
    "data_preparation": [
        "prepare_b2b.py",
        "prepare_dataco.py",
        "split_dataco.py",
    ],
    "data_validation": [
        "check_dataco_split.py",
        "validate_b2b_all_years.py",
        "validate_dataco.py",
    ],
    "model_training": [
        "save_dataco_model.py",
        "train_b2b_forecast.py",
        "train_baseline.py",
    ],
    "evaluation": [
        "evaluate_b2b_forecast.py",
        "evaluate_b2b_multiyear.py",
        "evaluate_b2b_seasonal.py",
        "evaluate_dataco.py",
        "test_saved_model.py",
    ],
}


def main():
    plan = []

    # Validate the entire move plan before moving anything.
    for folder, filenames in ORGANIZATION.items():
        destination_dir = SCRIPTS_DIR / folder

        for filename in filenames:
            source = ROOT / filename
            destination = destination_dir / filename

            if not source.exists():
                raise FileNotFoundError(
                    f"Expected script not found: {source}"
                )

            if destination.exists():
                raise FileExistsError(
                    f"Destination already exists: {destination}"
                )

            plan.append((source, destination))

    print(f"Scripts to move: {len(plan)}")
    print("\nPlanned folders:")

    for folder, filenames in ORGANIZATION.items():
        print(f"\n{folder}/")
        for filename in filenames:
            print(f"  {filename}")

    confirmation = input(
        "\nType MOVE to proceed, or anything else to cancel: "
    )

    if confirmation != "MOVE":
        print("Cancelled. No scripts were moved.")
        return

    moved = []

    try:
        for source, destination in plan:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(destination))
            moved.append((source, destination))

    except Exception:
        print("\nAn error occurred. Attempting to restore moved scripts.")

        for source, destination in reversed(moved):
            if destination.exists():
                source.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(destination), str(source))

        raise

    print(f"\nSuccessfully moved {len(moved)} scripts.")

    for folder in ORGANIZATION:
        print(f"  scripts/{folder}/")

    print("\nNo datasets or model artifacts were moved.")


if __name__ == "__main__":
    main()
