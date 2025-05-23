# streamlit run your_script_name.py
pip install streamlit pandas scikit-learn matplotlib seaborn numpy
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np # Required for np.array

# --- Page Configuration ---
st.set_page_config(
    page_title="AgriMind: Crop Recommendation AI",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Global Variables & Constants ---
DATA_PATH = 'Crop_recommendation.csv' # Make sure this CSV is in the same directory
FEATURES = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
TARGET = 'label'

# --- Caching Functions for Performance ---
@st.cache_data # Cache the data loading
def load_data(path):
    """Loads the crop recommendation dataset."""
    try:
        data = pd.read_csv(path)
        return data
    except FileNotFoundError:
        st.error(f"Error: The file '{path}' was not found. Please ensure it's in the same directory as the script.")
        return None

@st.cache_resource # Cache the model training process
def train_models_and_scaler(data):
    """Trains all models, selects the best one, and returns it along with the scaler and test data."""
    if data is None:
        return None, None, None, None, None, None

    X = data[FEATURES]
    y = data[TARGET]

    # Initialize and fit the scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    models_dict = {
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'SVM': SVC(random_state=42, probability=True), # Added probability=True for potential future use
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=200), # Increased max_iter
        'KNN': KNeighborsClassifier(),
        'Naive Bayes': GaussianNB()
    }

    trained_models = {}
    model_accuracies = {}
    best_model_name = None
    best_accuracy = 0.0
    best_model_obj = None

    for name, model in models_dict.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        
        trained_models[name] = model
        model_accuracies[name] = accuracy
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_name = name
            best_model_obj = model
            
    return best_model_obj, best_model_name, scaler, model_accuracies, X_test, y_test


# --- Load Data and Train Models ---
data_df = load_data(DATA_PATH)
if data_df is not None:
    best_model, best_model_name_global, scaler_global, model_accuracies_global, X_test_global, y_test_global = train_models_and_scaler(data_df)
else:
    st.sidebar.error("Dataset could not be loaded. App functionality is limited.")
    # Initialize to prevent errors later if data_df is None
    best_model, best_model_name_global, scaler_global, model_accuracies_global, X_test_global, y_test_global = None, None, None, {}, None, None


# --- Sidebar Navigation ---
st.sidebar.title("🌾 AgriMind Navigation")
app_mode = st.sidebar.selectbox(
    "Choose a Section",
    ["Crop Recommendation", "Exploratory Data Analysis (EDA)", "Model Performance"]
)

# --- Main Application Sections ---

if app_mode == "Crop Recommendation":
    st.title("🌱 Crop Recommendation Engine")
    st.markdown("""
    Welcome to the Crop Recommendation Engine! Adjust the sliders below to input the environmental and soil conditions. 
    The AI will then predict the most suitable crop for these conditions based on our trained model.
    """)

    if data_df is None or best_model is None or scaler_global is None:
        st.warning("The application is not fully initialized due to data loading issues. Prediction is unavailable.")
    else:
        st.subheader("Input Environmental and Soil Parameters:")
        
        col1, col2 = st.columns(2)

        with col1:
            N = st.slider("Nitrogen (N) content (kg/ha)", 0, 150, 90, help="Typical range: 0-150")
            P = st.slider("Phosphorus (P) content (kg/ha)", 0, 150, 42, help="Typical range: 0-150")
            K = st.slider("Potassium (K) content (kg/ha)", 0, 205, 43, help="Typical range: 0-205") # Max from dataset
        
        with col2:
            temperature = st.slider("Temperature (°C)", 0.0, 50.0, 20.9, step=0.1, help="Typical range: 0-50°C")
            humidity = st.slider("Relative Humidity (%)", 0.0, 100.0, 82.0, step=0.1, help="Typical range: 0-100%")
            ph = st.slider("Soil pH Level", 0.0, 14.0, 6.5, step=0.1, help="Typical range: 0-14")
        
        rainfall = st.slider("Rainfall (mm)", 0.0, 350.0, 202.9, step=0.1, help="Typical range: 0-350mm") # Max from dataset

        if st.button("🌿 Predict Suitable Crop", type="primary", use_container_width=True):
            with st.spinner("Analyzing conditions and predicting crop..."):
                input_features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
                input_scaled = scaler_global.transform(input_features)
                prediction = best_model.predict(input_scaled)
                
                st.success(f"**The AI recommends: __{prediction[0]}__**")
                st.balloons()

                # Displaying some info about the best model used
                st.markdown(f"*(Prediction made using the **{best_model_name_global}** model, which has an accuracy of **{model_accuracies_global.get(best_model_name_global, 0)*100:.2f}%** on the test set.)*")

elif app_mode == "Exploratory Data Analysis (EDA)":
    st.title("📊 Exploratory Data Analysis")
    st.markdown("Understanding the dataset used to train our crop recommendation model.")

    if data_df is None:
        st.warning("Dataset not loaded. EDA cannot be displayed.")
    else:
        st.subheader("Dataset Overview")
        st.dataframe(data_df.head())

        st.subheader("Dataset Information")
        # Capture data.info() output
        from io import StringIO
        buffer = StringIO()
        data_df.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)

        st.subheader("Statistical Summary")
        st.dataframe(data_df.describe())

        st.subheader("Crop Label Distribution")
        label_counts = data_df[TARGET].value_counts()
        st.bar_chart(label_counts)
        # Also show as a table
        st.write("Counts per crop label:")
        st.dataframe(label_counts)


        st.subheader("Feature Distributions (Histograms)")
        st.markdown("Histograms show the distribution of each numerical feature in the dataset.")
        
        # Create a figure for all histograms
        fig_hist, axes_hist = plt.subplots(4, 2, figsize=(15, 20)) # Adjusted for better layout
        axes_hist = axes_hist.flatten() # Flatten to easily iterate

        for i, feat in enumerate(FEATURES):
            if i < len(axes_hist): # Ensure we don't go out of bounds for axes
                sns.histplot(data_df[feat], color='skyblue', kde=True, ax=axes_hist[i])
                axes_hist[i].set_title(f'Distribution of {feat}', fontsize=12)
                axes_hist[i].grid(True)
        
        plt.tight_layout()
        st.pyplot(fig_hist)
        st.markdown("*KDE (Kernel Density Estimate) provides a smoothed version of the distribution.*")


        st.subheader("Feature Correlation Heatmap")
        st.markdown("""
        The heatmap below shows the Pearson correlation coefficients between different features. 
        Values close to 1 or -1 indicate a strong positive or negative linear correlation, respectively. 
        Values close to 0 indicate a weak linear correlation.
        """)
        
        # Calculate correlation matrix (excluding the target label for feature-feature correlation)
        correlation_matrix = data_df[FEATURES].corr()
        
        fig_corr, ax_corr = plt.subplots(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, cmap='viridis', fmt=".2f", linewidths=.5, ax=ax_corr)
        ax_corr.set_title('Correlation Between Different Features', fontsize=15)
        st.pyplot(fig_corr)

elif app_mode == "Model Performance":
    st.title("📈 Model Performance Evaluation")
    st.markdown("Comparison of different machine learning models trained on the dataset.")

    if data_df is None or not model_accuracies_global:
        st.warning("Dataset not loaded or models not trained. Performance metrics cannot be displayed.")
    else:
        st.subheader("Model Accuracies on Test Set")
        
        # Convert accuracies to DataFrame for better display and charting
        accuracies_df = pd.DataFrame(list(model_accuracies_global.items()), columns=['Model', 'Accuracy'])
        accuracies_df = accuracies_df.sort_values(by='Accuracy', ascending=False)
        
        # Display as a table
        st.dataframe(
            accuracies_df.style.format({"Accuracy": "{:.4f}"}).highlight_max(subset=['Accuracy'], color='lightgreen'),
            use_container_width=True
        )

        # Display as a bar chart
        fig_acc, ax_acc = plt.subplots(figsize=(10, 6))
        bars = sns.barplot(x='Accuracy', y='Model', data=accuracies_df, palette='mako', ax=ax_acc)
        ax_acc.set_xlabel("Accuracy Score", fontsize=12)
        ax_acc.set_ylabel("Model", fontsize=12)
        ax_acc.set_title("Model Comparison by Accuracy", fontsize=15)
        ax_acc.set_xlim(0, 1.05) # Set x-limit to just above 1.0 for full bars

        # Add accuracy values on bars
        for bar in bars.patches:
            ax_acc.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                        f'{bar.get_width()*100:.2f}%', 
                        va='center', ha='left', fontsize=10)

        st.pyplot(fig_acc)

        st.info(f"The best performing model is **{best_model_name_global}** with an accuracy of **{model_accuracies_global.get(best_model_name_global, 0)*100:.2f}%**.")
        
        # You could add classification reports here if desired
        # from sklearn.metrics import classification_report
        # if X_test_global is not None and y_test_global is not None and best_model is not None:
        #     st.subheader(f"Classification Report for {best_model_name_global}")
        #     test_predictions = best_model.predict(X_test_global)
        #     report = classification_report(y_test_global, test_predictions, output_dict=True)
        #     st.dataframe(pd.DataFrame(report).transpose())


# --- Footer ---
st.sidebar.markdown("---")
st.sidebar.info("🌾 AgriMind v1.0 | Your Smart Farming Assistant")
st.sidebar.markdown("Built with Streamlit & Scikit-learn")

# --- How to Run ---
# 1. Save this code as a Python file (e.g., app.py).
# 2. Make sure 'Crop_recommendation.csv' is in the same directory.
# 3. Open your terminal or command prompt.
# 4. Navigate to the directory where you saved the file.
# 5. Run the command: streamlit run app.py
