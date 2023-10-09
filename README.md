

# Cat-Dog Classifier with MobileNet using MLflow and Streamlit

## Introduction
This project demonstrates how to build a Cat-Dog Classifier using a MobileNet base model, train it, log the metrics and model to MLflow, register the model, and serve it using MLflow's model serving capabilities. 

## Architectur

![image](https://github.com/manas86/streamlit-mlflow/assets/30902765/7b4d0685-e179-41b6-8165-5f543d090c91)


## Dependencies
- Python 3.9
- TensorFlow/Keras
- MLflow
- matplotlib
- pydot
- graphviz
- requests

## Usage
1. Clone the repository and navigate to the project directory.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Download the training data and place it in the `./train-data` directory.
4. Run the training script `train.py` to train the Cat-Dog Classifier and log the results to MLflow.

```bash
python train.py
```

5. After training, you can use the MLflow UI to view the experiment and models.

## Model Deployment
### Serving the Model Locally
To serve the model locally, you can use the following command:

```bash
mlflow models serve --model-uri models:/cat_dog_classifier/production -p 5001 --no-conda
```

Once the model is served, you can make predictions by sending POST requests to `http://localhost:5001/invocations`.

### Transition to Production
To transition the model to production, use the following command:

```bash
mlflow models transition model-version-stage cat_dog_classifier 1 Production
```

This will make the model available in the `Production` stage for serving.

### Making Predictions
You can make predictions by sending POST requests to the model's serving endpoint. Here's an example using Python requests:

```python
import requests

endpoint = "http://localhost:5001/invocations"
image_request = {
    "instances": [your_image_array.tolist()]  # Replace with your image array
}

response = requests.post(endpoint, json=image_request)
predictions = eval(response.text)["predictions"]
print("This image is {:.2f}% cat and {:.2f}% dog.".format(
    100 * float(predictions[0][0]), 100 * float(predictions[0][1])))
```

Replace `[your_image_array.tolist()]` with the image data you want to classify.

## Conclusion
This project demonstrates how to train a Cat-Dog Classifier using MobileNet, log it to MLflow, register the model, and serve it locally for making predictions. You can further customize and expand this project for your specific use case.
Check `cat-dog-classifier.ipynb` file for more details. 

For more information, refer to the official documentation of [MLflow](https://mlflow.org/).

# Cat and Dog Image Classifier Streamlit App

## Introduction
This Streamlit app allows you to classify images as either a cat or a dog. It uses a pre-trained model to make predictions on uploaded images and displays the confidence levels for both cat and dog classifications.

## Dependencies
- Python 3.9
- Streamlit
- Pillow (PIL)
- utils (Custom utility functions)

You can install the required dependencies using `pip install -r requirements.txt`.

## Usage
1. Clone the repository and navigate to the project directory.
2. Ensure you have the required dependencies installed.
3. Run the Streamlit app using the following command:

```bash
streamlit run dashboard.py
```

4. Access the app in your web browser at the provided URL (typically, http://localhost:8501).

## App Pages
The app has three pages that you can navigate using the sidebar:

- **Predictor:** Upload an image of a cat or a dog to get predictions.
- **Transition:** Transition the model version to the "Production" stage.
- **Endpoint:** Make predictions using the model's serving endpoint.

### Predictor Page
- Upload an image by clicking on the "Upload cat or dog file" expander.
- The app will check if the uploaded file is in the correct format (png, jpg, jpeg).
- If the file format is correct, the app will save the image to a temporary directory and display it.
- Click the "Get predictions" button to fetch model predictions.
- Model predictions for cat and dog classifications will be displayed, along with confidence levels.

### Transition Page
- Transition the model version to the "Production" stage using this page.
- Ensure you have the correct model version selected before transitioning.

### Endpoint Page
- Make predictions using the model's serving endpoint.
- You can call the model endpoint programmatically using this page.

## Customization
You can customize and expand this app for your specific use case. The app uses a pre-trained model for image classification, and you can replace it with your own model if needed.

For more information on Streamlit, refer to the official documentation [here](https://docs.streamlit.io/).

## Acknowledgments
<img width="1497" alt="image" src="https://github.com/manas86/streamlit-mlflow/assets/30902765/c50dd402-5ccc-4848-a504-798d0b1a095e">
<img width="1494" alt="image" src="https://github.com/manas86/streamlit-mlflow/assets/30902765/82613acf-9c8c-45c2-93e0-634da4b8ccff">
<img width="1326" alt="image" src="https://github.com/manas86/streamlit-mlflow/assets/30902765/07eb3982-edfb-4b97-b288-b6660e90a359">
<img width="1483" alt="image" src="https://github.com/manas86/streamlit-mlflow/assets/30902765/abef9878-00e2-49dd-b098-cd33242814e1">
<img width="1489" alt="image" src="https://github.com/manas86/streamlit-mlflow/assets/30902765/dfa6adc2-cd35-4da6-8d09-7d9e36594387">
<img width="1503" alt="image" src="https://github.com/manas86/streamlit-mlflow/assets/30902765/f03c1533-9dd6-491e-80db-232309cb4ae7">
<img width="1368" alt="image" src="https://github.com/manas86/streamlit-mlflow/assets/30902765/53488825-0e6a-4607-b616-2ba2d788c9a2">
<img width="1354" alt="image" src="https://github.com/manas86/streamlit-mlflow/assets/30902765/241123c3-f551-4b68-9f73-8caa6cb2c342">
<img width="1398" alt="image" src="https://github.com/manas86/streamlit-mlflow/assets/30902765/ab3e752f-e199-49bd-ab38-470065d7e82e">
<img width="1407" alt="image" src="https://github.com/manas86/streamlit-mlflow/assets/30902765/8ba353b6-0924-48d2-b2d9-1c00d58daa02">
