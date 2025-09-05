import pandas as pd

class Preprocessor:
    def preprocess_input(self, data):
        data = self.handle_missing_values(data)
        data = self.encode_features(data)
        return data

    def handle_missing_values(self, df):
        return df.fillna("Unknown")

    def encode_features(self, df):
        return pd.get_dummies(df)