# Streamlit Linear Regression App (CRISP-DM)

This project contains a single-file Streamlit application (`app.py`) that demonstrates a Linear Regression problem following the **CRISP-DM** (Cross-Industry Standard Process for Data Mining) methodology.

## User Request Log
**You requested:**
> Build a single-file Streamlit app (app.py) using scikit-learn that demonstrates linear regression under the CRISP-DM workflow. Include UI sections for all six phases. Generate synthetic data: n∈[100,1000], x~Uniform(-100,100), a~Uniform(-10,10), b~Uniform(-50,50), noise~Normal(mean∈[-10,10], var∈[0,1000]); y=ax+b+noise. Add sidebar sliders for n, variance, seed, and a “Generate Data” button. Use train_test_split, StandardScaler, LinearRegression, MSE, RMSE, R². Plot scatter + regression line. Display true vs learned parameters. Provide prediction input and joblib saving. Output only clean, runnable code. Optimize layout, caching, and execution speed. Ensure compatibility with streamlit.app deployment. 幫我做這個 並且把我們的聊天紀錄整理成readme

## Implementation Details
We developed a well-structured `app.py` encompassing your requirements:

## 🌐 Live Demo & Deployment
We have set up the project so it can be deployed directly to Streamlit Community Cloud with a single click.

[![Deploy to Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/deploy?repository=andy19588/DIC7-ML&branch=main&mainModule=app.py)

**How to deploy:**
1. Click the "Deploy to Streamlit" button above.
2. Log in with your GitHub account when prompted.
3. Streamlit will automatically read the `app.py` and `requirements.txt` from your GitHub repository and build the environment.
4. Your Live Demo will be online in a few minutes, hosted on a beautiful `your-app-name.streamlit.app` link!

## Implementation Details

1. **Business Understanding**: Clearly defined the objective and success criteria for discovering the relationship between a single feature $X$ and the target $y$.
2. **Data Understanding**: Handled data generation based on the uniform and normal distribution constraints, adding a visual scatter plot to verify the generated variables.
3. **Data Preparation**: Leveraged `train_test_split` to divide the dataset and scaled features using `StandardScaler`.
4. **Modeling**: Employed scikit-learn's `LinearRegression`, showcasing a comparison of true parameters ($a$ and $b$) and the model's learned parameters appropriately transformed back to the original scale.
5. **Evaluation**: Computed and displayed MSE, RMSE, and $R^2$ metrics along with another plot laying the fitted regression line over the data points.
6. **Deployment**: Provided an input field for on-the-fly predictions and allowed users to download a `.joblib` payload embodying both the trained `.model` and `.scaler`.

Data caching was implemented via `@st.cache_data` making the app run predictably and smoothly upon interactions.

## How to Run

1. **Set Active Workspace:** I recommend keeping this directory (`C:\Users\user\.gemini\antigravity\scratch\streamlit_crisp_dm`) as your active workspace if you plan to continue extending it.
2. **Install requirements:**
   ```bash
   pip install streamlit numpy pandas matplotlib scikit-learn joblib
   ```
3. **Run the app:**
   ```bash
   streamlit run app.py
   ```

## AI Assistant Response (Cont.)
I have created your Streamlit app and organized the README as requested!

The files were placed in `C:\Users\user\.gemini\antigravity\scratch\streamlit_crisp_dm\`:
1. `app.py`: A clean, single-page Streamlit application implementing the CRISP-DM methodology.
2. `README.md`: Contains a tidy version of your prompt and instructions on how to use the app.

### How to test the app
First, you'll need the required packages installed in your environment if you haven't done so already. You can install them by running:
```bash
pip install streamlit numpy pandas matplotlib scikit-learn joblib
```

Then, you can simply run the streamlit app using:
```bash
streamlit run app.py
```

Let me know if you would like me to adjust any aesthetics or tweak the math logic behind the random generations!
