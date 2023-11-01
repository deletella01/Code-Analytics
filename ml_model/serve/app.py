import streamlit as st
import pickle
import time
import warnings
import pandas as pd
from pathlib import Path

# Ignore warnings
warnings.filterwarnings('ignore')

# Configure Streamlit app settings
st.set_page_config(layout="wide",
                   page_title="CITES Species Classifier",
                   page_icon="https://forestrypedia.com/wp-content/uploads/2018/05/Biodiversity-1.png"
                   )

# Hide the default "Made with Streamlit" footer
hide_default_format = """
       <style>
       footer {visibility: hidden;}
       </style>
       """
st.markdown(hide_default_format, unsafe_allow_html=True)

# Custom CSS to add an image to the title
custom_css = """
<style>
.title-container {
    display: flex;
    align-items: center;
}

.title-text {
    margin-left: 20px;
}
</style>
"""

# Use the custom CSS to style the title
st.markdown(custom_css, unsafe_allow_html=True)

# Create a title container with an image and text
st.markdown('<div class="title-container"><img src="https://forestrypedia.com/wp-content/uploads/2018/05/Biodiversity-1.png" alt="Image" width="200"/><h1 class="title-text">CITES Species Classifier</h1></div>', unsafe_allow_html=True)

with st.container():
   st.info("""**Project Summary for the CITES Species Classifier App**

Welcome to the "CITES Species Classifier" – an app designed to explore and predict the CITES appendix classification of wildlife species. Here's what you need to know:

**CITES Appendices Explained:**
- **Appendix I:** High protection for species facing extinction.
- **Appendix II:** Species not threatened with extinction but require trade control.
- **Appendix III:** Species regulated by member states.

**What You Can Do:**
- **Explore Categories:** Learn about species in each appendix.
- **Predict Appendix:** Test your knowledge with our classifier.
- **Learn About Conservation:** Understand why these classifications matter.

**Why It Matters:**
- Protecting biodiversity and combating illegal wildlife trade is vital.

Explore, predict, and join the conversation on species conservation. Welcome to the CITES Species Classifier app!""")
   
# Use the 'st.radio' function to create a radio button widget with the given options.
# The user can select one option from the radio button.
prediction_type = st.radio("Select Prediction Type", options=["Single Prediction", "Batch Prediction"])

# Use the 'st.write' function to display a message indicating the user's selection.
# This message shows what the user has selected as the prediction type.
st.warning(f"You have selected: {prediction_type}")

# Specify model path
# This is for deployment
model_path = Path('premiere_project/serve/model/model_2023-11-01T18:58:43.960271.pkl')

# This is for local runs
# model_path = 'model/model_2023-11-01T18:58:43.960271.pkl'

# Load the model from the file
with open(model_path, 'rb') as file:
    loaded_model = pickle.load(file)

@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

def main():
    """
    Main function to handle user interaction and predictions for the CITES Species Classifier app.

    This function allows users to input data either for a single prediction or batch prediction, and it
    displays the predictions accordingly.

    Parameters:
        None

    Returns:
        None
    """
    if prediction_type == 'Single Prediction':
        # If the user selects 'Single Prediction', gather input features from the sidebar
        with st.sidebar:
            st.header('Input Features')
            taxon = st.text_input("TAXON")
            Class = st.text_input("CLASS")
            order = st.text_input("ORDER")
            family = st.text_input("FAMILY")
            genus = st.text_input("GENUS")
            term = st.text_input("TERM")
            purpose = st.text_input("PURPOSE")
            source = st.text_input("SOURCE")
            predict = st.button("Predict Class")
        
        # Create a dictionary from the input features
        input_dict = {"taxon": taxon,
                      "class": Class,
                      "order": order,
                      "family": family,
                      "genus": genus,
                      "term": term,
                      "purpose": purpose,
                      "source": source}
        
        # Create a DataFrame from the input dictionary and rename columns
        input_df = pd.DataFrame(input_dict, index=[0])
        input_df = input_df.rename(columns={
            "taxon": "Taxon",
            "class": "Class",
            "order": "Order",
            "family": "Family",
            "genus": "Genus",
            "term": "Term",
            "purpose": "Purpose",
            "source": "Source"})
        
        if predict:
            with st.spinner('Running inference...'):
                time.sleep(5)
            # Make predictions using the loaded model and display the result
            prediction = loaded_model.predict(input_df)
            st.success('Inference Done!')
            st.subheader("Prediction")
            st.text(f"This species belongs in class: {prediction}")

    if prediction_type == 'Batch Prediction':
        # If the user selects 'Batch Prediction', allow them to upload a CSV file
        input_file = st.file_uploader("Upload your CSV file")
        if input_file:
            input_df = pd.read_csv(input_file)
            with st.spinner('Running inference...'):
                time.sleep(100)
            # Make predictions for the batch and append them to the DataFrame
            predictions = loaded_model.predict(input_df)
            st.success('Inference Done!')
            with st.spinner('Appending Predictions to DataFrame...'):
                time.sleep(30)
                input_df["Predicted Class"] = predictions

            st.dataframe(input_df, use_container_width=True)
            # st.download_button(
            #     label="Download CSV",
            #     data=convert_df(input_df),
            #     file_name='scored_df.csv',
            #     mime='text/csv',
            # )

if __name__ == "__main__":

    main()