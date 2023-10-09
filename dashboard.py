import streamlit as st
import os
from PIL import Image
from utils import get_model_predictions
import tempfile

# ----------- General things
st.title('cat and dog image classifier')

# ----------- Sidebar
page = st.sidebar.selectbox('Page Navigation', ["Predictor", "Transition", "Endpoint"])


upload_columns = st.columns([2, 1])
# File upload
file_upload = upload_columns[0].expander(label="Upload cat or dog file")
uploaded_file = file_upload.file_uploader("Choose cat or dog file",
                                          type=['png', 'jpg', 'jpeg'],
                                          help="please insert an image of cat or dog in png, jpg or jpeg format.")
if uploaded_file is not None:
    filename, file_extension = os.path.splitext(uploaded_file.name)

    if file_extension not in ['.png', '.jpg', '.jpeg']:
        st.error(f'File type is not {filename} and {file_extension}')
    else:
        st.success("File type is right format")
        with tempfile.TemporaryDirectory() as temp_dir:
            # read uploaded image as byte
            image_bytes = uploaded_file.read()
            # save the image to temporary directory
            temp_image_path = os.path.join(temp_dir, f"temp_image{file_extension}")
            with open(temp_image_path, "wb") as temp_image_file:
                temp_image_file.write(image_bytes)

            image = Image.open(temp_image_path)
            # st.image(image)
            st.write(f"Image saved to: {temp_image_path}")
            upload_columns[1].image(image)
            submit = upload_columns[1].button("Get predictions")

            # ----------- Submission
            st.markdown("""---""")
            if submit:
                with st.spinner(text="Fetching model prediction..."):
                    # Call model endpoint
                    prediction = get_model_predictions(temp_image_path, page)

                    # ----------- Ouputs
                    outputs = st.columns([2, 1])
                    outputs[0].markdown("Prediction: ")
                    prediction_details = st.expander(label="Model details")
                    details = prediction_details.columns([2, 1])
                    details[0].markdown("Cat Confidence: ")
                    details[0].markdown("Dog Confidence: ")
                    details[1].markdown("{:.2f}%".format(100 * float(prediction[0][0])))
                    details[1].markdown("{:.2f}%".format(100 * float(prediction[0][1])))
                    # st.write("This image is {:.2f}% cat and {:.2f}% dog.".format(100 * float(prediction[0][0]),
                    #                                                             100 * float(prediction[0][1])))


