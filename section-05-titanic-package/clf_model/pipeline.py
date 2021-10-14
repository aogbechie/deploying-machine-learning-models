from feature_engine.encoding import OneHotEncoder, RareLabelEncoder
from feature_engine.imputation import (
    AddMissingIndicator,
    CategoricalImputer,
    MeanMedianImputer,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from clf_model.config.core import config
from clf_model.processing import features as pp

titanic_pipe = Pipeline(
    [
        # ===== IMPUTATION =====
        # impute categorical variables with string 'missing'
        (
            "categorical_imputation",
            CategoricalImputer(
                imputation_method="missing",
                variables=config.model_config.categorical_features,
            ),
        ),
        # add missing indicator to numerical variables
        (
            "missing_indicator",
            AddMissingIndicator(variables=config.model_config.numerical_features),
        ),
        # impute numerical variables with the median
        (
            "median_imputation",
            MeanMedianImputer(
                imputation_method="median",
                variables=config.model_config.numerical_features,
            ),
        ),
        # Extract first letter from cabin
        (
            "extract_letter",
            pp.ExtractLetterTransformer(
                variables=config.model_config.extract_letter_features
            ),
        ),
        # == CATEGORICAL ENCODING ======
        # remove categories present in less than 5% of the observations (0.05)
        # group them in one category called 'Rare'
        (
            "rare_label_encoder",
            RareLabelEncoder(
                tol=config.model_config.tol,
                n_categories=1,
                replace_with="Rare",
                variables=config.model_config.categorical_features,
            ),
        ),
        # encode categorical variables using one hot encoding into k-1 variables
        (
            "categorical_encoder",
            OneHotEncoder(
                drop_last=True, variables=config.model_config.categorical_features
            ),
        ),
        # scale using standardization
        ("scaler", StandardScaler()),
        # logistic regression (use C=0.0005 and random_state=0)
        (
            "Logit",
            LogisticRegression(
                C=config.model_config.C, random_state=config.model_config.random_state
            ),
        ),
    ]
)
