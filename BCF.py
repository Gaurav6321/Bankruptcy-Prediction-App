import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Set the color scheme
COLOR_SCHEME = {
    'background': '#f4f4f4',
    'text': '#000000',
    'header': '#2f4f4f',
    'button': '#008080',
}

st.set_page_config(
    page_title="Bankruptcy Prediction App",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS styling for the app
st.markdown(
    f"""
    <style>
        .reportview-container {{
            background-color: {COLOR_SCHEME['background']};
            color: {COLOR_SCHEME['text']};
        }}
        .sidebar .sidebar-content {{
            background-color: {COLOR_SCHEME['header']};
            color: {COLOR_SCHEME['text']};
        }}
        .streamlit-button {{
            background-color: {COLOR_SCHEME['button']};
            color: {COLOR_SCHEME['text']};
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

def remove_outliers(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

def feature_engineering(data):
    # Add your feature engineering steps here
    # Example: data['new_feature'] = data['feature1'] * data['feature2']
    return data

def load_data():
    data = pd.read_csv(r"/workspaces/Bankruptcy_Prediction_App/.devcontainer/Bankruptcy Prediction.csv")

    # Remove outliers from numerical columns
    numerical_columns = data.select_dtypes(include='float').columns
    for col in numerical_columns:
        data = remove_outliers(data, col)

    # Perform feature engineering
    data = feature_engineering(data)

    return data

def train_model(X_train, y_train, X_test, y_test):
    # Define a smaller parameter grid for faster tuning
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [None, 10],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    # Create a RandomForestClassifier instance
    rf_model = RandomForestClassifier(random_state=42)

    # Use GridSearchCV for hyperparameter tuning
    grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, scoring='accuracy', cv=5)
    grid_search.fit(X_train, y_train)

    # Get the best parameters and train the final model
    best_params = grid_search.best_params_
    final_model = RandomForestClassifier(random_state=42, **best_params)
    final_model.fit(X_train, y_train)

    # Display the best parameters found by GridSearchCV
    st.subheader("Best Hyperparameters:")
    st.write(best_params)

    # Display model evaluation metrics
    st.subheader("Model Evaluation Metrics:")
    st.write(f"Training Accuracy: {accuracy_score(y_train, final_model.predict(X_train)):.2%}")
    st.write(f"Testing Accuracy: {accuracy_score(y_test, final_model.predict(X_test)):.2%}")
    # Replace the confusion matrix line with the following:
    st.write("Confusion Matrix:")
    st.write(confusion_matrix(y_test, final_model.predict(X_test), labels=y_test.unique()))

    st.write("Classification Report:")
    st.write(classification_report(y_test, final_model.predict(X_test)))

    return final_model

def main():
    # Set the title of your Streamlit app
    st.title('Bankruptcy Prediction App')

    # Load data
    df = load_data()

    # Display the target variable
    st.write("Target Variable:", "Bankrupt?")

    # Replace 'target_column' with the actual target variable in your dataset
    target_column = 'Bankrupt?'

    if target_column not in df.columns:
        st.error(f"Target column '{target_column}' not found in the dataset.")
        st.stop()

    # Assuming 'target_column' is the target variable
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = train_model(X_train, y_train, X_test, y_test)

    # Display user input for prediction
    st.sidebar.header('User Input:')

    features = {}
    for feature in X.columns:
        features[feature] = st.number_input(feature, min_value=df[feature].min(), max_value=df[feature].max())

    submitted = st.sidebar.button("Predict")

    # Check if the button is clicked
    if submitted:
        # Convert user input to a DataFrame
        user_input = pd.DataFrame([features])

        # Predict bankruptcy on user input
        prediction = model.predict(user_input)
        probability = model.predict_proba(user_input)[:, 0]

        # Display prediction result
        st.subheader('Prediction Result:')
        st.write(f"The model predicts that the company is {'Bankrupt' if prediction[0] == 1 else 'Not Bankrupt'}")
        st.write(f"Probability of Bankruptcy: {probability[0]:.2%}")

if _name_ == '_main_':
    main()
