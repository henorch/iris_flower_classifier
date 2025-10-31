import streamlit as st
import joblib
import numpy as np

# Load the model
model = joblib.load('./model/knn_model.pkl')

st.title("🌸 Iris Flower Classification")
st.write("Enter your flower features separated by commas, e.g. `5.1,3.5,1.4,0.2`")

features = st.text_input("Enter features (sepal length, sepal width, petal length, petal width)")

if st.button('Predict'):
    try:
        # Convert input to numeric array
        feat = np.array([float(x.strip()) for x in features.split(',')]).reshape(1, -1)

        # Predict probabilities
        chance = model.predict_proba(feat)[0]

        # Predict the final class
        prediction = model.predict(feat)[0]

        # Map numeric prediction to species name
        if prediction == 0:
            output = "Setosa"
        elif prediction == 1:
            output = "Versicolor"
        else:
            output = "Virginica"

        # Show probabilities clearly
        st.subheader("Prediction Probabilities")
        st.write(f"Setosa: {chance[0]*100:.2f}%")
        st.write(f"Versicolor: {chance[1]*100:.2f}%")
        st.write(f"Virginica: {chance[2]*100:.2f}%")

        # Display the most likely species
        st.success(f"🌼 Predicted Iris species: **{output}**")

    except Exception as e:
        st.error(f"⚠️ Invalid input! Please enter 4 numeric values separated by commas.\n\nError: {e}")
