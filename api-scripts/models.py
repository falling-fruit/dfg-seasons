"""
This file contains the models' class structure that will be used to make predictions.
"""

from datetime import date
import random


class SampleModel:
    """
    This is a sample model that will be used to make predictions.
    """

    def __init__(self):
        pass

    def predict(self, location: str, date: date):
        """
        This method will be used to make predictions.
        """
        return random.random()
