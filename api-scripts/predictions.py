"""
This file contains the predict function which takes in a type, location, and date
and returns a prediction value.

The predict function uses the geopy library to geocode the location and then uses
the model to make a prediction.
"""

from datetime import date
from geopy.geocoders import Nominatim
import joblib


# Using Nominatim Api
geolocator = Nominatim(user_agent="fallingfruit")


def predict(type: str, location: str, date: date):
    """
    This function takes in a type, location, and date and returns a prediction value.
    """

    # check if date is in the past
    if date < date.today():
        raise Exception("Invalid date. Date is in the past.")

    # Using geocode()
    geocode_location = geolocator.geocode(location)

    # check if location is valid
    if not geocode_location:
        raise Exception("Invalid location")

    # Loading model
    try:
        type = type.lower()
        model = joblib.load(f"./models/{type}_model.pkl")
        prediction = model.predict(geocode_location, date)
    except FileNotFoundError:
        raise Exception("Invalid type")

    # Displaying result details
    print(f"Type: {type}")
    print(f"Location: {geocode_location}")
    print(f"Prediction Value: {prediction}")
    return prediction


if __name__ == "__main__":
    predict("apple", "20500 USA", date(2024, 1, 1))
