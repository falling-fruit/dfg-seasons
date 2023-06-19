"""
This file contains the predict function which takes in a type, location, and date
and returns a prediction value.
"""

from datetime import date
import joblib


def predict(type: str, location: tuple[float, float], date: date):
    """
    This function takes in a type, location, and date and returns a prediction value.
    """

    # check if date is in the future
    if date <= date.today():
        raise Exception("Invalid date. Date is in the past.")

    # Loading model
    try:
        type = type.lower()
        model = joblib.load(f"./models/{type}_model.pkl")
        prediction = model.predict(location, date)
    except FileNotFoundError:
        raise Exception("Invalid type")

    # Displaying result details
    print(f"Type: {type}")
    print(f"Latitude Longitude: {location}")
    print(f"Prediction Value: {prediction}")
    return prediction


if __name__ == "__main__":
    predict("apple", (40.712776, -74.005974), date(2024, 1, 1))
