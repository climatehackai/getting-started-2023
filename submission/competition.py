import os
import sys
from typing import Any, Generator

import h5py
import numpy as np


class BaseEvaluator:
    def __init__(self) -> None:
        self.setup()

    def setup(self) -> None:
        """Sets up anything required for evaluation, e.g. loading a model."""
        pass

    def predict(self, features: h5py.File) -> Generator[np.ndarray, Any, None]:
        """Makes solar PV predictions for a test set.

        Args:
            features (h5py.File): Solar PV, satellite imagery, weather forecast and air quality forecast features.

        Yields:
            Generator[np.ndarray, Any, None]: A batch of predictions.
        """

        raise NotImplementedError

    def batch(
        self, features: h5py.File, variables: list[str], batch_size: int = 32
    ) -> Generator[list[np.ndarray], Any, None]:
        """Batches up test set input features for the data variables requested.

        Args:
            features (h5py.File): _description_
            batch_size (int, optional): _description_. Defaults to 32.

        Yields:
            Generator[list[np.ndarray], Any, None]: _description_
        """

        if not variables:
            raise ValueError("At least one data variable must be specified.")

        for variable in variables:
            assert variable in features

        datasets = {variable: features[variable] for variable in variables}
        for i in range(0, len(datasets[variables[0]]), batch_size):  # type: ignore
            yield [datasets[variable][i : i + batch_size] for variable in variables]  # type: ignore

    def evaluate(self) -> None:
        """Communicates with and handles your evaluation on the DOXA AI platform."""

        stream_directory = os.environ.get("DOXA_STREAMS", ".")
        with open(f"{stream_directory}/out", "w") as f:
            f.write("OK\n")
            f.flush()

            for batch in self.predict(features=h5py.File(sys.argv[1], "r")):
                for sample in batch:
                    assert sample.shape == (48,)
                    f.write(",".join([str(float(number)) for number in sample]) + "\n")

                f.flush()
