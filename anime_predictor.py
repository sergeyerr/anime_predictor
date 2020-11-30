from catboost import CatBoostClassifier
def anime_predict(book, english, twoch, games, vegan, path = "final_model"):
    classifier = CatBoostClassifier()
    classifier.load_model("final_model")
    return classifier.predict([book, english, twoch, games, vegan])