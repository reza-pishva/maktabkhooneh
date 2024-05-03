import streamlit as st
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
import joblib

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Train the KNN classifier (as shown in the previous code snippet)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)

# # Save the model (you can do this in the previous script)
# joblib.dump(knn, 'knn_model.joblib')

# Load the model within the Streamlit script
knn_loaded = joblib.load('knn_model.joblib')

# Streamlit app
st.title('Iris Classification with KNN')

# Input bars for the features
sepal_length = st.number_input('Sepal Length', min_value=0.0, max_value=10.0, value=5.0)
sepal_width = st.number_input('Sepal Width', min_value=0.0, max_value=10.0, value=3.0)
petal_length = st.number_input('Petal Length', min_value=0.0, max_value=10.0, value=4.0)
petal_width = st.number_input('Petal Width', min_value=0.0, max_value=10.0, value=1.0)

# When 'Predict' is clicked, make the prediction and display it
if st.button('Predict'):
    prediction = knn_loaded.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    st.write(f'The predicted class is: {iris.target_names[prediction][0]}')
