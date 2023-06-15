"""
Dump models to pickle files.
"""

import joblib
from models import SampleModel

if __name__ == "__main__":
    # sample_types = ["apple", "banana", "cherry", "durian", "eggplant"]

    model = SampleModel()
    joblib.dump(model, "./models/apple_model.pkl")

    model = SampleModel()
    joblib.dump(model, "./models/banana_model.pkl")

    model = SampleModel()
    joblib.dump(model, "./models/cherry_model.pkl")

    model = SampleModel()
    joblib.dump(model, "./models/durian_model.pkl")

    model = SampleModel()
    joblib.dump(model, "./models/eggplant_model.pkl")
