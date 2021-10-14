import typing as t

import numpy as np
import pandas as pd

from clf_model import __version__ as _version
from clf_model.config.core import config
from clf_model.processing.data_manager import load_pipeline
from clf_model.processing.validation import validate_inputs

pipeline_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
_titanic_pipe = load_pipeline(file_name=pipeline_file_name)


def make_prediction(
    *,
    input_data: t.Union[pd.DataFrame, dict],
) -> dict:
    """Make a prediction using a saved model pipeline."""

    data = pd.DataFrame(input_data)
    validated_data, errors = validate_inputs(input_data=data)
    results = {"predictions_class": None, "predictions_proba": None,
      "version": _version, "errors": errors}

    if not errors:
        predictions_class = _titanic_pipe.predict(
            X=validated_data[config.model_config.features]
        )
        predictions_proba = _titanic_pipe.predict_proba(
            X=validated_data[config.model_config.features]
        )[:,-1]

        results = {
            "predictions_class": predictions_class,
            "predictions_proba": predictions_proba,
            "version": _version,
            "errors": errors,
        }

    return results