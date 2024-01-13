import os

import h5py
import numpy as np

from submission.run import Evaluator

DATA_PATH = "data/validation/data.hdf5"


def main():
    # Load the data (combined features & targets)
    try:
        data = h5py.File(DATA_PATH, "r")
    except FileNotFoundError:
        print(f"Unable to load features at `{DATA_PATH}`")
        return

    # Switch into the submission directory
    cwd = os.getcwd()
    os.chdir("submission")

    # Make predictions on the data
    try:
        evaluator = Evaluator()

        predictions = []
        for batch in evaluator.predict(features=data):
            assert batch.shape[-1] == 48
            predictions.append(batch)
    finally:
        os.chdir(cwd)

    # Output the mean absolute error
    mae = np.mean(np.absolute(data["targets"] - np.concatenate(predictions)))
    print("MAE:", mae)


if __name__ == "__main__":
    main()
