import io
import mlflow
import requests
import mlflow.keras
from tensorflow import keras
import tensorflow as tf
import json
import requests

mlflow.set_tracking_uri("http://localhost:5000")
input_shape = (224, 224, 3)

def get_model_predictions(uploaded_file, page):
    """
    Get model predictions
    ENDPOINT = Calls an endpoint to get the predictions, for page = Endpoint
    REGISTRY = Loads model from registry and predicts for page = Predictor
    PRODUCTION = Loads the model from production stage for page = Transition
    """

    if page == "Predictor":
        model_name = "cat_dog_classifier"
        model_version = 1

        loaded_model = mlflow.keras.load_model(model_uri=f"models:/{model_name}/{model_version}")
        # input_image = keras.preprocessing.image.load_img(io.BytesIO(uploaded_file.read()), target_size=input_shape)

        input_image = keras.preprocessing.image.load_img(uploaded_file, target_size=input_shape)
        img_array = keras.preprocessing.image.img_to_array(input_image)
        img_array = tf.expand_dims(img_array, 0)

        predictions = loaded_model.predict(img_array)

    if page == "Transition":
        model_name = "cat_dog_classifier"
        model_stage = "production"

        loaded_model = mlflow.keras.load_model(model_uri=f"models:/{model_name}/{model_stage}")
        input_image = keras.preprocessing.image.load_img(uploaded_file, target_size=input_shape)
        img_array = keras.preprocessing.image.img_to_array(input_image)
        img_array = tf.expand_dims(img_array, 0)

        predictions = loaded_model.predict(img_array)

    if page == "Endpoint":
        DEPLOYED_ENDPOINT = "http://127.0.0.1:5001/invocations"
        headers = {"Content-Type":"application/json"}
        input_image = keras.preprocessing.image.load_img(uploaded_file, target_size=input_shape)
        img_array = keras.preprocessing.image.img_to_array(input_image)
        img_array = tf.expand_dims(img_array, 0)

        response = requests.post(url=DEPLOYED_ENDPOINT,
                                 json={"instances": img_array.numpy().tolist()})
        predictions = eval(response.text)["predictions"]

    return predictions
