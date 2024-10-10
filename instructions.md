# Instructions to Recreate the Iris Flowers Classification App

## Step 1: Set Up the Project Directory

1. **Create a New Directory**: Create a new directory on your local machine for the project.
    ```bash
    mkdir iris_classification_app
    ```

2. **Navigate to the Project Directory**: Change your current directory to the newly created project directory.
    ```bash
    cd iris_classification_app
    ```

## Step 2: Initialize the Project

1. **Initialize a Git Repository**: Initialize a new Git repository in the project directory.
    ```bash
    git init
    ```

2. **Create a Virtual Environment**: Create a virtual environment to manage dependencies.
    ```bash
    python -m venv venv
    ```

3. **Activate the Virtual Environment**: Activate the virtual environment.
    - On Windows:
        ```bash
        venv\Scripts\activate
        ```
    - On macOS/Linux:
        ```bash
        source venv/bin/activate
        ```

## Step 3: Install Dependencies

1. **Install Required Libraries**: Install the necessary dependencies using pip.
    ```bash
    pip install streamlit pandas scikit-learn
    ```

2. **Create a `requirements.txt` File**: Create a `requirements.txt` file in the project directory and list the installed dependencies.
    ```bash
    pip freeze > requirements.txt
    ```

## Step 4: Create the Application Files

1. **Create the Main Application File**: Create a file named `app.py` in the project directory.
    ```bash
    touch app.py
    ```

2. **Create a Model File**: Create a file named `model.py` to handle the machine learning model.
    ```bash
    touch model.py
    ```

## Step 5: Implement the Machine Learning Model

1. **Import Libraries in `model.py`**: In `model.py`, import the necessary libraries.
    ```python
    import pandas as pd
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score
    ```

2. **Load and Prepare the Iris Dataset**: Write code to load the Iris dataset and prepare it for training.
    ```python
    # Load the Iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    ```

3. **Train the SVC Model**: Write code to train the Support Vector Classifier (SVC) model.
    ```python
    # Train the SVC model
    model = SVC(probability=True)
    model.fit(X_train, y_train)
    ```

4. **Save the Trained Model**: Write code to save the trained model to a file.
    ```python
    import joblib
    joblib.dump(model, 'iris_model.pkl')
    ```

## Step 6: Implement the Streamlit Application

1. **Import Libraries in `app.py`**: In `app.py`, import the necessary libraries.
    ```python
    import streamlit as st
    import pandas as pd
    import joblib
    ```

2. **Load the Trained Model**: Write code to load the trained model from the file.
    ```python
    model = joblib.load('iris_model.pkl')
    ```

3. **Create the User Interface**: Write code to create the user interface using Streamlit.
    ```python
    st.title('Iris Flowers Classification App')

    # Input features using sliders
    sepal_length = st.slider('Sepal Length', 0.0, 10.0, 5.0)
    sepal_width = st.slider('Sepal Width', 0.0, 10.0, 5.0)
    petal_length = st.slider('Petal Length', 0.0, 10.0, 5.0)
    petal_width = st.slider('Petal Width', 0.0, 10.0, 5.0)

    # Create a DataFrame with the input features
    input_data = pd.DataFrame({
        'sepal_length': [sepal_length],
        'sepal_width': [sepal_width],
        'petal_length': [petal_length],
        'petal_width': [petal_width]
    })
    ```

4. **Make Predictions**: Write code to make predictions using the loaded model.
    ```python
    if st.button('Predict'):
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)

        st.write('Predicted Iris Species:', iris.target_names[prediction][0])
        st.write('Prediction Probabilities:', prediction_proba)
    ```

## Step 7: Run the Application

1. **Run the Streamlit Application**: Start the Streamlit application by running the following command in the project directory.
    ```bash
    streamlit run app.py
    ```

2. **Access the Application**: Open your web browser and navigate to the URL provided by Streamlit (usually `http://localhost:8501`).

## Step 8: Test the Application

1. **Input Features**: Use the sliders to input the sepal length, sepal width, petal length, and petal width.

2. **Predict**: Click the "Predict" button to see the predicted iris species and the prediction probabilities.

## Conclusion

By following these steps, you should be able to recreate the Iris Flowers Classification App from scratch. This application demonstrates the integration of a machine learning model with a web application using Streamlit and scikit-learn.