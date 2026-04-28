import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import io

st.set_page_config(page_title="CRISP-DM Linear Regression", layout="wide")

# Sidebar
st.sidebar.title("Data Generation Settings")
n_samples = st.sidebar.slider("Number of samples (n)", min_value=100, max_value=1000, value=500)
noise_var = st.sidebar.slider("Noise Variance", min_value=0, max_value=1000, value=100)
random_seed = st.sidebar.slider("Random Seed", min_value=0, max_value=100, value=42)

generate_btn = st.sidebar.button("Generate Data")

@st.cache_data
def generate_data(n, var, seed):
    np.random.seed(seed)
    x = np.random.uniform(-100, 100, n)
    a = np.random.uniform(-10, 10)
    b = np.random.uniform(-50, 50)
    
    mean_noise = np.random.uniform(-10, 10)
    std_noise = np.sqrt(var)
    noise = np.random.normal(mean_noise, std_noise, n)
    
    y = a * x + b + noise
    return pd.DataFrame({'X': x, 'y': y}), a, b

if "data" not in st.session_state or generate_btn:
    data, true_a, true_b = generate_data(n_samples, noise_var, random_seed)
    st.session_state.data = data
    st.session_state.true_a = true_a
    st.session_state.true_b = true_b

st.title("Linear Regression under CRISP-DM")
st.markdown("This Streamlit app demonstrates a simple Linear Regression problem using the **CRISP-DM** (Cross-Industry Standard Process for Data Mining) methodology.")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "1. Business Understanding",
    "2. Data Understanding",
    "3. Data Preparation",
    "4. Modeling",
    "5. Evaluation",
    "6. Deployment"
])

with tab1:
    st.header("1. Business Understanding")
    st.markdown("""
    **Objective**: Build a predictive model to understand the linear relationship between a feature ($X$) and a target variable ($y$).
    
    **Success Criteria**: Achieve a high $R^2$ score and low RMSE, indicating the model can accurately capture the underlying trend despite noise.
    """)

with tab2:
    st.header("2. Data Understanding")
    st.write("We generate synthetic data based on a linear equation $y = ax + b + noise$.")
    
    data = st.session_state.data
    st.dataframe(data.head())
    
    st.subheader("Data Distribution")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(data['X'], data['y'], c='blue', alpha=0.5, label='Data points')
    ax.set_xlabel('X')
    ax.set_ylabel('y')
    ax.set_title("Scatter Plot of X vs y")
    ax.legend()
    st.pyplot(fig)

with tab3:
    st.header("3. Data Preparation")
    st.markdown("In this phase, we split the data into training and testing sets, and scale the features.")
    
    X = data[['X']]
    y = data['y']
    
    test_size = st.slider("Test Set Size", 0.1, 0.5, 0.2)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed)
    
    st.write(f"Training set: {X_train.shape[0]} samples")
    st.write(f"Testing set: {X_test.shape[0]} samples")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    st.session_state.scaler = scaler
    st.session_state.X_train_scaled = X_train_scaled
    st.session_state.X_test_scaled = X_test_scaled
    st.session_state.y_train = y_train
    st.session_state.y_test = y_test
    st.success("Data successfully split and scaled.")

with tab4:
    st.header("4. Modeling")
    st.markdown("We will train a Linear Regression model on the prepared data.")
    
    model = LinearRegression()
    model.fit(st.session_state.X_train_scaled, st.session_state.y_train)
    
    st.session_state.model = model
    
    # Calculate original scale parameters
    scale_factor = model.coef_[0] / st.session_state.scaler.scale_[0]
    intercept = model.intercept_ - (scale_factor * st.session_state.scaler.mean_[0])
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("True Parameters")
        st.write(f"Slope (a): **{st.session_state.true_a:.4f}**")
        st.write(f"Intercept (b): **{st.session_state.true_b:.4f}**")
        
    with col2:
        st.subheader("Learned Parameters (Original Scale)")
        st.write(f"Slope (a): **{scale_factor:.4f}**")
        st.write(f"Intercept (b): **{intercept:.4f}**")
    
    st.success("Model successfully trained.")

with tab5:
    st.header("5. Evaluation")
    
    model = st.session_state.model
    X_test_scaled = st.session_state.X_test_scaled
    y_test = st.session_state.y_test
    
    y_pred = model.predict(X_test_scaled)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    st.write(f"**Mean Squared Error (MSE)**: {mse:.4f}")
    st.write(f"**Root Mean Squared Error (RMSE)**: {rmse:.4f}")
    st.write(f"**R² Score**: {r2:.4f}")
    
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.scatter(st.session_state.data['X'], st.session_state.data['y'], color='blue', alpha=0.3, label='Actual Data')
    
    # Line
    x_range = np.linspace(st.session_state.data['X'].min(), st.session_state.data['X'].max(), 100).reshape(-1, 1)
    x_range_scaled = st.session_state.scaler.transform(pd.DataFrame(x_range, columns=['X']))
    y_line = model.predict(x_range_scaled)
    
    ax2.plot(x_range, y_line, color='red', linewidth=2, label='Regression Line')
    ax2.set_xlabel('X')
    ax2.set_ylabel('y')
    ax2.set_title("Regression Fit Line")
    ax2.legend()
    st.pyplot(fig2)

with tab6:
    st.header("6. Deployment")
    st.markdown("Use the trained model to make new predictions or download it for external usage.")
    
    pred_input = st.number_input("Enter a value for X to predict y:", value=0.0)
    
    if st.button("Predict"):
        pred_scaled = st.session_state.scaler.transform(pd.DataFrame([[pred_input]], columns=['X']))
        pred_y = st.session_state.model.predict(pred_scaled)[0]
        st.write(f"Predicted y for X={pred_input} is: **{pred_y:.4f}**")
    
    st.subheader("Export Model")
    export_dict = {
        'model': st.session_state.model,
        'scaler': st.session_state.scaler
    }
    
    # Save directly to buffer
    buffer = io.BytesIO()
    joblib.dump(export_dict, buffer)
    buffer.seek(0)
    
    st.download_button(
        label="Download Trained Model (joblib)",
        data=buffer,
        file_name="linear_regression_model.joblib",
        mime="application/octet-stream"
    )
