import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier

class TargetEncoder:
    def __init__(self, encode_map=None):
        if encode_map is None:
            self.encode_map = {"Canceled": 0, "Not_Canceled": 1}
        else:
            self.encode_map = encode_map
        self.inverse_map = {v: k for k, v in self.encode_map.items()}

    def transform(self, y):
        return y.replace(self.encode_map)

    def inverse_transform(self, y_encoded):
        return y_encoded.replace(self.inverse_map)

class DataHandler:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.input_df = None
        self.output_df = None

    def load_data(self):
        self.data = pd.read_csv(self.file_path)
        self.data.drop(['Booking_ID'], axis=1, inplace=True)

    def handle_missing_values(self):
        self.data['avg_price_per_room'].fillna(99.9, inplace=True)
        self.data['type_of_meal_plan'].fillna('Meal Plan 1', inplace=True)
        self.data['required_car_parking_space'].fillna(0.0, inplace=True)

    def create_input_output(self, target_column):
        self.output_df = self.data[target_column]
        self.input_df = self.data.drop(target_column, axis=1)

class ModelHandler:
    def __init__(self, input_data, output_data):
        self.input_data = input_data
        self.output_data = output_data
        self.encode_inputs()
        self.label_encode_target()
        self.split_data()
        self.create_model()

    def encode_inputs(self):
        self.meal_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        self.room_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        self.market_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')

        meal_encoded = self.meal_encoder.fit_transform(self.input_data[['type_of_meal_plan']])
        room_encoded = self.room_encoder.fit_transform(self.input_data[['room_type_reserved']])
        market_encoded = self.market_encoder.fit_transform(self.input_data[['market_segment_type']])

        meal_df = pd.DataFrame(meal_encoded, columns=self.meal_encoder.get_feature_names_out())
        room_df = pd.DataFrame(room_encoded, columns=self.room_encoder.get_feature_names_out())
        market_df = pd.DataFrame(market_encoded, columns=self.market_encoder.get_feature_names_out())

        self.input_data.reset_index(drop=True, inplace=True)
        self.input_data = pd.concat([self.input_data, meal_df, room_df, market_df], axis=1)
        self.input_data.drop(['type_of_meal_plan', 'room_type_reserved', 'market_segment_type'], axis=1, inplace=True)

    def label_encode_target(self):
        self.target_encoder = TargetEncoder()
        self.output_data = self.target_encoder.transform(self.output_data)

    def split_data(self, test_size=0.2, random_state=42):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.input_data, self.output_data, test_size=test_size, random_state=random_state)

    def create_model(self):
        self.model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

    def train_model(self):
        self.model.fit(self.x_train, self.y_train)

    def evaluate_model(self):
        predictions = self.model.predict(self.x_test)
        print(f"Accuracy: {accuracy_score(self.y_test, predictions):.4f}")
        print("Classification Report:\n", classification_report(self.y_test, predictions))
        print("Confusion Matrix:\n", confusion_matrix(self.y_test, predictions))

    def make_prediction(self):
        self.y_predict = self.model.predict(self.x_test)

    def create_report(self):
        print("\nClassification Report\n")
        print(classification_report(self.y_test, self.y_predict))

    def save_model_to_file(self):
        pickle.dump(self.model, open('XGB_booking.pkl', 'wb'))
        pickle.dump(self.target_encoder, open('booking_encode.pkl', 'wb'))  # Save custom encoder
        pickle.dump(self.meal_encoder, open('oneHot_encode_type.pkl', 'wb'))
        pickle.dump(self.room_encoder, open('oneHot_encode_room.pkl', 'wb'))
        pickle.dump(self.market_encoder, open('oneHot_encode_market.pkl', 'wb'))

if __name__ == '__main__':
    data_handler = DataHandler('Dataset_B_hotel.csv')
    data_handler.load_data()
    data_handler.handle_missing_values()
    data_handler.create_input_output('booking_status')

    model_handler = ModelHandler(data_handler.input_df, data_handler.output_df)
    model_handler.train_model()
    model_handler.evaluate_model()
    model_handler.make_prediction()
    model_handler.create_report()
    model_handler.save_model_to_file()