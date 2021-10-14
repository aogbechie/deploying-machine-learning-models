from clf_model.config.core import config
from clf_model.processing.features import ExtractLetterTransformer


def test_extraction_of_first_letter_transformer(sample_input_data):
    # Given
    transformer = ExtractLetterTransformer(
        variables=config.model_config.extract_letter_features  # ['cabin']
    )
    assert sample_input_data["cabin"].iat[414] == "C105"

    # When
    subject = transformer.fit_transform(sample_input_data)
    # Then
    assert subject["cabin"].iat[414] == "C"
