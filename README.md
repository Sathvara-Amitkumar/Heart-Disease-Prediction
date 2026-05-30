# Heart Disease Prediction System

A simple machine learning app that predicts whether a person is likely to have heart disease based on their health data.

## What this project does

- Uses patient features such as age, blood pressure, cholesterol, maximum heart rate, ECG results, and chest pain type.
- Applies a trained K-Nearest Neighbors model to estimate the likelihood of heart disease.
- Provides a Streamlit web interface for easy input and prediction.

## How to use

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
3. Enter the patient details in the form.
4. Click **Predict** to see whether the model estimates a low or high chance of heart disease.

## Files included

- `app.py` - Streamlit application for user input and prediction.
- `heart.csv` - dataset used for training and exploration.
- `heart_diases.ipynb` - notebook with data analysis, preprocessing, and model training.
- `KNN_heart.pkl`, `scaler.pkl`, `columns.pkl` - saved model and preprocessing artifacts.

## Notes

- The app is intended for demonstration only and not as medical advice.
- The model prediction depends on the quality and completeness of the input values.
