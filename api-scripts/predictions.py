## This is the home script for the predictions class

# TODO: work on input error handling (ex. invalid model, invalid paths, invalid months, dates, etc)
# this will be lots of testing. 

import pickle

class Predictor:

    # mostly used for testing
    model_object = None
    weather_object = None

    # two optsions for initializing:

    # from model / weather paths
    def from_paths(self, model_path, weather_path):
        self.model_object = self.load_model(model_path)
        self.weather_object = self.load_weather(weather_path)

    # passing objects directly
    def from_objects(self, model_object, weather_object):
        self.model_object = model_object
        self.weather_object = weather_object

    @staticmethod
    def info():
        return "Version 0.1, by Peter Benson and Andrew Suh"

    @staticmethod
    def help():
        pass

    def load_model(self, model_path):
         # assuming pickled weather data
        model_pickle = open(model_path, 'rb')

        sample_model_object = pickle.load(model_pickle)

        return sample_model_object
    
    def load_weather(self, weather_path):
        # assuming pickled weather data
        weather_pickle = open(weather_path, 'rb')

        loaded_weather_object = pickle.load(weather_pickle)

        return loaded_weather_object

    def format_date(self, date):
        
        month = date
        # this is bad, change later

        return month
    
    @staticmethod
    def month_to_name(month):
        months = [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December"
        ]

        return months[month]

    def predict(self, location, date, species, seasonality_binary=False, output_dict=False):

        # Format date into the proper format – we only need months
        # this might change if we need more granular, or if we also need years
        month = self.format_date(date)

        weather_data = self.get_historical_weather(location, month)

        final_seasonality_prediction = self.run_model(weather_data, month, species)

        if seasonality_binary:
            final_seasonality_prediction = round(final_seasonality_prediction)

        # handle logic for returning a binary "is in season or not", or a raw prediction
        if output_dict:
            return {
                "result": final_seasonality_prediction,
                "confidence_interval": 0.6,
                "weather_data_volume": 200,
            }

        else:
            return final_seasonality_prediction

    def run_model(self, weather, month, species):
        # access trained model here using the trained model path 
        
        return self.model_object.get_prediction(month, species)
    
    def get_historical_weather(self, location, month):

        return self.weather_object.get_temperature(location, month)

class Model:

    model_array = {
        "apple": [0, 0, 0, 0, 0, 0, 0.25, 0.6, 0.75, 0.9, 0.7, 0.2]
    }

    def get_prediction(self, month, species):
        return self.model_array[species][month]


class WeatherData:

    # temperatures will be in Celsius

    weather_array = {
        "boston": [-1.2, -0.1, 3.5, 9.2, 14.7, 20.0, 23.4, 22.6, 18.7, 12.7, 7.1, 2.1]
    }

    def get_temperature(self, location, month):
        return self.weather_array[location][month]


### –––––––– TESTING SECTION –––––––––

test_model = Model()
test_weather = WeatherData()

test_predict = Predictor()
test_predict.from_objects(test_model, test_weather)

for i in range(12):
    print(test_predict.predict("boston", i, "apple", True))