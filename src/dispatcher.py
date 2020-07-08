from sklearn import ensemble
from catboost import CatBoostClassifier

# explicitly require this experimental feature
from sklearn.experimental import enable_hist_gradient_boosting  # noqa

# now you can import normally from ensemble
from sklearn.ensemble import HistGradientBoostingClassifier


# catb = CatBoostClassifier()
# catb.fit(X_train, y_train)
# catb.score(X_test, y_test))

# https://github.com/catboost/tutorials/blob/master/python_tutorial.ipynb


# 0.75091
MODELS = {
    "randomforest": ensemble.RandomForestClassifier(
        n_estimators=100, n_jobs=-1, verbose=2
    ),
    "extratrees": ensemble.ExtraTreesClassifier(n_estimators=200, n_jobs=-1, verbose=2),
    "catboost": CatBoostClassifier(),
    "histboost": HistGradientBoostingClassifier(),
}
