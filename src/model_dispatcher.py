from sklearn import linear_model, naive_bayes, ensemble, svm

MODELS = {
    "logistic_regression": linear_model.LogisticRegression(max_iter=1000, random_state=42),
    "naive_bayes": naive_bayes.MultinomialNB(),
    "random_forest": ensemble.RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    "svm": svm.SVC(C=10)
}