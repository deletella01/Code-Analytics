# Wildlife Trade Risk Prediction
## Group: Code Analytics
This project studies Biodiversity Preservation - for conservation, and ecosystem health. Our project studies the risk involved in trading each species, and we developed a model to predict how risky it is to trade a certain species. We aim to analyze the international wildlife trade data from CITES in 2016-2017 and derive actionable insights contributing to biodiversity preservation and conservation efforts.

### Research Development 
#### Data Preprocessing
The data was prepared using several procedures. Features with over 30,000 non-null values were dropped, and null values in the Import and Export features were filled with zero. Features with null values beyond a threshold of 30,000 were removed, including "Year", "Origin", and "Unit". Leaky features were also dropped. Exploratory data analysis (EDA) revealed missing values in 90% of Units, 62% of origin, and other features.

#### Model Development
The first model was developed by dropping features with large numbers of null values, filling null values with zero, and encoding categorical features. The model was trained using Logistic Regression, Random Forest, and Decision Tree Classifiers, achieving accuracy scores of 88%, 97%, and 98%, respectively. A hard voting classifier was used to combine the models, achieving an accuracy of 91% on the test dataset and 93% on the training dataset.

#### Model Pipeline
The final model used a pipeline with an ordinal encoder, RobustScaler, and a voting classifier. The data was cleaned and prepared using a function called 'data_wrangler'. The model was trained and achieved 90%, 99%, and 54% precision scores for Appendix classes I, II, and III, respectively.

#### Deployment
The model was deployed as a Streamlit application with two modes: single prediction and batch prediction. The app allows users to enter values or upload a CSV file to receive predictions. The app was pushed to a GitHub repo and deployed on the Streamlit cloud, where it can be accessed via a specified URL.

### Conclusion
Our analysis shows that a lot of wildlife Species trade is detrimental
to Biodiversity Preservation and increases the rate at which a lot of species can be lost to extinction due to human activities. We have built a web application that predicts the level of trade risk as a result of a user’s input on the features of the trade.

Our web application link can be found **[here](https://wildlifeclassifier.streamlit.app/)**.

## Authors:
- **Bamidele Tella**
Email: deletella01@gmail.com
- **Bharat Kumar Jhawar**
Email: jhawarbharat52@gmail.com
- **Okerinde Peculiar Temilola**
Email: temilolaokerinde@gmail.com
- **Olorunleke White**
Email: lekewhite@gmail.com
- **Priscila Waihiga Kamiri**
Email: priscilla.waihiga@gmail.com
- **George Israel Opeyemi**
Email: georgeisrael18@outlook.com
- **Halimah Oyebanji**
Email: halimahoyebanji@gmail.com
- **Lukman Aliyu**
Email: Lukman.aliyu.adejoh@gmail.com
- **Oladimeji Williams**
Email: mathematicianf@gmail.com
- **Martha Edet**
Email: marthavictoredet@gmail.com
- **Duke Effiom**
Email: effiomduke@gmail.com
