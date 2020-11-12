from sklearn import linear_model, naive_bayes

MODELS = {
    "logistic_regression": linear_model.LogisticRegression(max_iter=1000, random_state=42),
    "naive_bayes": naive_bayes.MultinomialNB()
}