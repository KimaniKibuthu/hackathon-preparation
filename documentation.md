# Iris Flowers Classification App

## Overview

The Iris Flowers Classification App is a Streamlit-based web application designed to predict the species of an iris flower based on input features such as sepal length, sepal width, petal length, and petal width. The application leverages a pre-trained Support Vector Classifier (SVC) model to make predictions and provides a user-friendly interface for inputting the necessary features and displaying the predicted species.

## Purpose

The primary purpose of this application is to demonstrate the use of machine learning models in a web application context. It serves as an educational tool for understanding how to integrate machine learning models with web frameworks like Streamlit. Additionally, it provides a practical example of how to use the Iris dataset from scikit-learn for classification tasks.

## Features

- **User Interface**: A simple and intuitive interface built with Streamlit.
- **Input Features**: Users can input sepal length, sepal width, petal length, and petal width using sliders.
- **Prediction**: The application uses a pre-trained SVC model to predict the iris species based on the input features.
- **Display**: The predicted iris species and prediction probabilities are displayed to the user.

## Dependencies

The application relies on the following dependencies:

- **Streamlit**: A framework for creating web applications with Python.
- **Pandas**: A data manipulation and analysis library.
- **Scikit-learn**: A machine learning library that provides the Iris dataset and the SVC model.

To install the necessary dependencies, you can use the following command:

```bash
pip install streamlit pandas scikit-learn
```

## Installation Instructions

1. **Clone the Repository**: Clone the repository to your local machine using the following command:

    ```bash
    git clone <repository_url>
    ```

2. **Navigate to the Project Directory**: Change your current directory to the project directory:

    ```bash
    cd cloned_repo
    ```

3. **Install Dependencies**: Install the required dependencies by running:

    ```bash
    pip install -r requirements.txt
    ```

    Note: Ensure you have a `requirements.txt` file in the project directory listing the necessary dependencies.

## Usage

1. **Run the Application**: Start the Streamlit application by running the following command in the project directory:

    ```bash
    streamlit run app.py
    ```

2. **Access the Application**: Open your web browser and navigate to the URL provided by Streamlit (usually `http://localhost:8501`).

3. **Input Features**: Use the sliders to input the sepal length, sepal width, petal length, and petal width.

4. **Predict**: Click the "Predict" button to see the predicted iris species and the prediction probabilities.

## Conclusion

The Iris Flowers Classification App is a straightforward yet powerful example of integrating machine learning models with web applications. It serves as a valuable resource for learning and experimenting with Streamlit and scikit-learn. For further development, consider exploring additional features, improving the user interface, or integrating more advanced machine learning models.

---

This documentation provides a comprehensive overview of the project, its purpose, features, dependencies, installation instructions, and usage. It is designed to be informative and easy to follow, ensuring that new developers or users can understand and utilize the project effectively.