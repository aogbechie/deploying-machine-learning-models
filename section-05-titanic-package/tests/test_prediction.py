import math

import numpy as np

from clf_model.predict import make_prediction


def test_make_prediction(sample_input_data):
    # Given
    expected_first_prediction_value = 0
    expected_first_prediction_proba = 0.33527866
    expected_no_predictions = 418

    # When
    result = make_prediction(input_data=sample_input_data)

    # Then
    predictions_class = result.get("predictions_class")
    predictions_proba = result.get("predictions_proba")
    assert isinstance(predictions_class, list)
    assert isinstance(predictions_proba, list)
    assert isinstance(predictions_class[0], np.int64)
    assert isinstance(predictions_proba[0], np.float64)
    assert result.get("errors") is None
    assert len(predictions_class) == expected_no_predictions
    assert math.isclose(
        predictions_class[0], expected_first_prediction_value, abs_tol=0
    )
    assert math.isclose(
        predictions_proba[0], expected_first_prediction_proba, abs_tol=0.005
    )
