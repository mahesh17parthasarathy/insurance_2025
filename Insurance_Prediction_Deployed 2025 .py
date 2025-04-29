import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go  # For radar chart
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from pandas_profiling import ProfileReport
import streamlit.components.v1 as components
from ydata_profiling import ProfileReport
# Title of the App
st.markdown("""
    <h1 style='text-align: center; color: #FF5733; font-family: Verdana, sans-serif; font-size: 40px;'>
        'Insurance Charges Prediction Using Linear Regression'
    </h1>
""", unsafe_allow_html=True)
col1, col2, col3 = st.columns([1, 2, 1])
with col2:  # Center column
    st.image("/content/img.jpg", caption="Insurance charges prediction", width=300)
# File uploader for users to upload their own dataset
uploaded_file = st.file_uploader('Upload your insurance data CSV', type='csv')
# Load the dataset
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    # Display dataset information
    st.write('Data Preview:')
    st.dataframe(df.head(), use_container_width=True)
    st.data_editor(df, use_container_width=True)
    st.write('Data Information:')
    st.write(df.info())
    # Check for missing values
    st.write('Missing Values:')
    st.write(df.isnull().sum())
    if st.button("Generate EDA Report"):
      profile = ProfileReport(df, explorative=True)
      profile_path = "eda_report.html"
      profile.to_file(profile_path)
      # Display report in Streamlit
      with open(profile_path, "r", encoding="utf-8") as f:
          html_content = f.read()
      components.html(html_content, height=800, scrolling=True)
    # Handling missing values using imputation (mean for numerical, most frequent for categorical)
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    cat_cols = df.select_dtypes(include=['object']).columns
    # Imputation strategy for numerical columns
    num_imputer = SimpleImputer(strategy='mean')
    df[num_cols] = num_imputer.fit_transform(df[num_cols])
    # Imputation strategy for categorical columns
    cat_imputer = SimpleImputer(strategy='most_frequent')
    df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])
    st.write('Data after handling missing values:')
    st.dataframe(df.head())
    # Label encode categorical variables
    le_smoker = LabelEncoder()
    le_region = LabelEncoder()
    df['smoker'] = le_smoker.fit_transform(df['smoker'])
    df['region'] = le_region.fit_transform(df['region'])
    # Selecting the relevant features and target variable
    X = df[['age', 'bmi', 'children', 'smoker', 'region']]  # Features
    y = df['charges']  # Target variable
    # Train-Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Train Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    # Prediction on test set
    y_pred = model.predict(X_test)
    R2 = model.score(X_test, y_test)
    st.write(f"Model Test R2 Score: {R2:.2f}")
    mse = mean_squared_error(y_test, y_pred)
    st.write(f"Model Test MSE: {mse:.2f}")
    # --- User Input for Prediction ---
    st.header('Predict Insurance Charges')
    # Collect user input
    age = st.number_input('Age', min_value=18, max_value=100, value=30)
    bmi = st.number_input('BMI', min_value=15.0, max_value=40.0, value=25.0)
    children = st.number_input('Number of Children', min_value=0, max_value=5, value=0)
    smoker = st.selectbox('Smoker', ['yes', 'no'])
    region = st.selectbox('Region', ['southeast', 'southwest', 'northeast', 'northwest'])
    # Encode user input using the same encoders
    smoker_encoded = le_smoker.transform([smoker])[0]
    if region in le_region.classes_:
      region_encoded = le_region.transform([region])[0]
    else:
      st.error("Selected region was not seen during training. Please select a valid region.")
      region_encoded = None  # Handle this case appropriately
    if region_encoded is not None:
      # Combine user input into a dataframe
      input_data = pd.DataFrame([[age, bmi, children, smoker_encoded, region_encoded]],
                              columns=['age', 'bmi', 'children', 'smoker', 'region'])
      # Predict the charges using the trained model
      prediction = model.predict(input_data)
      st.subheader(f'Predicted Insurance Charges: {prediction[0]:.2f}')
    # --- Add Radar Chart ---
    st.header('Radar Chart of User Input Compared to Dataset')
    # Get min and max values of the features from the dataset for comparison
    max_values = X.max()
    min_values = X.min()
    # Normalize user input values based on the dataset's feature range
    input_values = np.array([age, bmi, children, smoker_encoded, region_encoded])
    # Normalize input values between 0 and 1
    normalized_input = (input_values - min_values) / (max_values - min_values)
    # Define the radar chart categories
    categories = ['Age', 'BMI', 'Children', 'Smoker', 'Region']
    # Define the radar chart
    fig = go.Figure()
    # Add trace for normalized user input
    fig.add_trace(go.Scatterpolar(
        r=normalized_input,
        theta=categories,
        fill='toself',
        name='User Input'
    ))
    # Add trace for the maximum values from the dataset
    fig.add_trace(go.Scatterpolar(
        r=np.ones_like(input_values),
        theta=categories,
        fill='none',
        name='Max Dataset Value',
        line=dict(color='red', dash='dash')
    ))
    # Add trace for the minimum values from the dataset
    fig.add_trace(go.Scatterpolar(
        r=np.zeros_like(input_values),
        theta=categories,
        fill='none',
        name='Min Dataset Value',
        line=dict(color='blue', dash='dash')
    ))
    # Update radar chart layout
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1])
        ),
        showlegend=True
    )
    # Display the radar chart
    st.plotly_chart(fig)

