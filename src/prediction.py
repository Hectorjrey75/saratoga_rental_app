import joblib

def load_model(file_path):
    model = joblib.load(file_path)
    return model

def predict(model, data):
    predictions = model.predict(data)
    return predictions