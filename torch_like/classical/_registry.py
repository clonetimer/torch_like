"""Static scikit-learn estimator catalog.

The catalog contains import paths only so importing :mod:`torch_like` does not
pull scikit-learn into memory. Returned estimator classes are the original
scikit-learn classes; torch_like does not wrap or emulate their fit/predict API.
"""

from __future__ import annotations

ESTIMATOR_SPECS: dict[str, dict[str, object]] = {
    # Linear models
    "LinearRegression": {
        "module": "sklearn.linear_model",
        "kind": "regression",
        "family": "linear_model",
    },
    "Ridge": {
        "module": "sklearn.linear_model",
        "kind": "regression",
        "family": "linear_model",
    },
    "LogisticRegression": {
        "module": "sklearn.linear_model",
        "kind": "classification",
        "family": "linear_model",
    },
    "SGDClassifier": {
        "module": "sklearn.linear_model",
        "kind": "classification",
        "family": "linear_model",
    },
    "SGDRegressor": {
        "module": "sklearn.linear_model",
        "kind": "regression",
        "family": "linear_model",
    },
    # Trees / ensembles
    "DecisionTreeClassifier": {
        "module": "sklearn.tree",
        "kind": "classification",
        "family": "tree",
    },
    "DecisionTreeRegressor": {
        "module": "sklearn.tree",
        "kind": "regression",
        "family": "tree",
    },
    "RandomForestClassifier": {
        "module": "sklearn.ensemble",
        "kind": "classification",
        "family": "ensemble",
    },
    "RandomForestRegressor": {
        "module": "sklearn.ensemble",
        "kind": "regression",
        "family": "ensemble",
    },
    "ExtraTreesClassifier": {
        "module": "sklearn.ensemble",
        "kind": "classification",
        "family": "ensemble",
    },
    "ExtraTreesRegressor": {
        "module": "sklearn.ensemble",
        "kind": "regression",
        "family": "ensemble",
    },
    "GradientBoostingClassifier": {
        "module": "sklearn.ensemble",
        "kind": "classification",
        "family": "ensemble",
    },
    "GradientBoostingRegressor": {
        "module": "sklearn.ensemble",
        "kind": "regression",
        "family": "ensemble",
    },
    "HistGradientBoostingClassifier": {
        "module": "sklearn.ensemble",
        "kind": "classification",
        "family": "ensemble",
    },
    "HistGradientBoostingRegressor": {
        "module": "sklearn.ensemble",
        "kind": "regression",
        "family": "ensemble",
    },
    "IsolationForest": {
        "module": "sklearn.ensemble",
        "kind": "anomaly_detection",
        "family": "ensemble",
    },
    # Kernel / neighbor methods
    "SVC": {
        "module": "sklearn.svm",
        "kind": "classification",
        "family": "kernel",
    },
    "SVR": {
        "module": "sklearn.svm",
        "kind": "regression",
        "family": "kernel",
    },
    "KNeighborsClassifier": {
        "module": "sklearn.neighbors",
        "kind": "classification",
        "family": "neighbors",
    },
    "KNeighborsRegressor": {
        "module": "sklearn.neighbors",
        "kind": "regression",
        "family": "neighbors",
    },
    # Unsupervised utilities commonly used alongside the model library.
    "KMeans": {
        "module": "sklearn.cluster",
        "kind": "clustering",
        "family": "cluster",
    },
    "DBSCAN": {
        "module": "sklearn.cluster",
        "kind": "clustering",
        "family": "cluster",
    },
    "PCA": {
        "module": "sklearn.decomposition",
        "kind": "decomposition",
        "family": "decomposition",
    },
}
