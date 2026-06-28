"""
Classical machine learning classifiers.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import xgboost as xgb


def get_classifier(name: str, seed: int = 42):
    """
    name: "rf" | "lgbm" | "xgb" | "logreg"
    """
    classifiers = {
        "rf": RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=seed),
        "lgbm": lgb.LGBMClassifier(n_estimators=300, random_state=seed, verbose=-1),
        "xgb": xgb.XGBClassifier(n_estimators=300, random_state=seed, verbosity=0),
        "logreg": LogisticRegression(max_iter=1000, random_state=seed),
    }
    return classifiers[name]
