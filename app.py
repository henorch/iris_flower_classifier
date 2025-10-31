import streamlit as st
import joblib
import matplotlib.pyplot as plt
import numpy as np
import wikipedia

# Load the model
model = joblib.load('./model/knn_model.pkl')

st.title("🌸 Iris Flower Classification")
st.write("Enter your flower features separated by commas, e.g. `5.1,3.5,1.4,0.2`")

features = st.text_input("Enter features (sepal length, sepal width, petal length, petal width)")

if st.button('Predict'):
    try:
        # Convert input to numeric array
        feat = np.array([float(x.strip()) for x in features.split(',')]).reshape(1, -1)

        # Predict probabilities and class
        chance = model.predict_proba(feat)[0]
        prediction = model.predict(feat)[0]

        # Map numeric prediction to species name
        species = ["Setosa", "Versicolor", "Virginica"]
        output = species[prediction]

        # --- Display probabilities ---
        st.subheader("Prediction Probabilities")
        for name, prob in zip(species, chance):
            st.write(f"{name}: {prob * 100:.2f}%")

        

        # Wikipedia lookup
        flower_name = f"Iris {output.capitalize()}"
        with st.spinner("🔍 Please wait, getting flower information..."):
            summary = wikipedia.summary(flower_name, sentences=2)
            image = wikipedia.page(flower_name).images[0]

        # Display result
        st.success(f"🌼 Predicted Iris species: **{output}**")

        col1, col2 = st.columns([1, 3])  # image smaller column
        with col1:
            st.image(image, width=120, caption=flower_name)
        with col2:
            st.markdown(f"###  {flower_name}")
            st.write(summary)

    except Exception as e:
        st.warning("Couldn't fetch online description. Displaying basic info instead.")
        st.write(f"The predicted species is **{output}**.")
        st.write(f"(Error: {e})")
