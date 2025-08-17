# Deep Autoregressive Time Series Forecasting for GOOG Stock Price

## Problem Statement and Goal of Project

Predicting stock market movements is a notoriously complex challenge due to the inherent volatility and non-stationary nature of financial data. This project addresses this challenge by developing and evaluating a sophisticated deep learning model to forecast the daily closing price of Google's stock (GOOG).

The primary goal is to implement a Deep Autoregressive (DeepAR) model using an LSTM architecture to produce probabilistic forecasts, providing not just a single point estimate but a distribution of potential future values. This project serves as a practical demonstration of applying modern time series forecasting techniques, robust model training practices, and systematic evaluation methodologies.

## Solution Approach

My approach is a comprehensive, end-to-end process that begins with data acquisition and ends with rigorous model evaluation and optimization.

1.  **Data Acquisition and Preparation:**

      * Historical daily stock data for GOOG from January 2015 to June 2025 was downloaded using the `yfinance` library.
      * The raw data was loaded into a Pandas DataFrame, where I performed initial cleaning and wrangling to flatten the multi-index columns for easier manipulation.
      * To handle non-trading days (weekends, holidays), I converted the dataset into a `Darts TimeSeries` object, filling missing dates and interpolating values to ensure a continuous daily frequency.

2.  **Feature Engineering (Covariates):**

      * **Past Covariates:** To provide the model with more context about recent market dynamics, I engineered several features based on historical closing prices, including a 7-day moving average (`ma_7`), 7-day moving standard deviation (`std_7`), and 1-day rate of change (`roc_1`).
      * **Future Covariates:** Since time-based patterns (e.g., day of the week, month of the year) are known in advance, I created one-hot encoded features for the day, month, and weekday. These serve as future covariates that help the model capture seasonality.

3.  **Data Scaling and Splitting:**

      * Both the target time series and the covariates were scaled to a [0, 1] range to stabilize the training process and improve model convergence.
      * The complete dataset was strategically split into training (70%), validation (15%), and test (15%) sets to ensure robust evaluation.

4.  **Model Architecture and Training:**

      * I implemented a DeepAR-style model using the `RNNModel` (LSTM cell) from the Darts library, incorporating a `GaussianLikelihood` function. This enables the model to output a probability distribution, allowing for the generation of confidence intervals around the point forecast.
      * The model was trained using PyTorch Lightning, leveraging several advanced callbacks for a professional workflow:
          * `EarlyStopping`: To prevent overfitting by halting training when validation loss stops improving.
          * `ModelCheckpoint`: To automatically save the best-performing model based on validation loss.
          * `ReduceLROnPlateau`: To dynamically adjust the learning rate for more efficient convergence.
          * `RichProgressBar` and `Timer`: To enhance the monitoring of the training process.

5.  **Evaluation and Optimization:**

      * **Performance Metrics:** The model was evaluated on the test set using Mean Absolute Error (MAE) and Mean Absolute Percentage Error (MAPE).
      * **Backtesting:** I performed `historical_forecasts` to simulate the model's performance on historical data under realistic, expanding-window conditions, providing a more reliable measure of its real-world utility.
      * **Hyperparameter Tuning:** A comprehensive grid search was set up to systematically explore various model hyperparameters (e.g., sequence length, hidden dimensions, layers, dropout) and identify the optimal configuration.
      * **Architectural Exploration:** To demonstrate a broader understanding, I also included the setup for an `NBEATSModel`, another state-of-the-art architecture, for future comparison and benchmarking.

## Technologies & Libraries

  * **Core Language:** Python
  * **Time Series & Deep Learning:** Darts, PyTorch, PyTorch Lightning
  * **Data Handling:** Pandas, NumPy
  * **Data Acquisition:** yfinance
  * **Data Preprocessing:** scikit-learn
  * **Visualization:** Matplotlib

## Description about Dataset

The dataset consists of historical daily stock price data for Google (Ticker: **GOOG**), sourced from **Yahoo Finance**.

  * **Time Period:** January 1, 2015 – June 13, 2025
  * **Frequency:** Daily
  * **Features:** `Open`, `High`, `Low`, `Close`, `Adj Close`, and `Volume`.
  * **Target Variable:** The `Close` price was used as the target for forecasting.

## Installation & Execution Guide

To replicate this project, please follow these steps:

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```
2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Run the Notebook:**
    Launch Jupyter Notebook or JupyterLab and open the `Deep Autoregressive_02.ipynb` file to execute the code.

## Key Results / Performance

The model was evaluated using two primary methods, yielding solid results that demonstrate its predictive capabilities.

**1. Test Set Prediction:**

  * **Test MAE:** 0.1693
  * **Test MAPE:** 20.81%

**2. Backtesting (Historical Forecasts):**
This more rigorous evaluation simulates real-world performance over multiple historical windows.

  * **MAE:** 0.0936
  * **MAPE:** 12.00%
  * **sMAPE:** 13.15%
  * **RMSE:** 0.0998

The lower error rates in the backtesting scenario suggest that the model generalizes well and can adapt effectively over time.

## Sample Output

The plot below visualizes the model's forecast on the unseen test data. The shaded area represents the 90% confidence interval (from the 5th to the 95th quantile), showcasing the probabilistic nature of the DeepAR forecast.

## Additional Learnings / Reflections

This project was an excellent opportunity to deepen my understanding of advanced time series forecasting. While some models were experimental, they were intentionally included to demonstrate my learning process and analytical skills.

  * **Probabilistic Forecasting:** Implementing the `GaussianLikelihood` function was a key learning. It shifted the objective from predicting a single point to estimating a probability distribution, which is far more valuable for risk assessment in financial applications.
  * **Robust Training Workflow:** Using PyTorch Lightning with callbacks like `EarlyStopping` and `ModelCheckpoint` reinforced the importance of a structured and automated training process. It not only prevents overfitting but also ensures that the best version of the model is always preserved.
  * **Feature Engineering Impact:** The creation of both past and future covariates highlighted their significant impact on model performance. Understanding which features can be used and when (past vs. future) is critical for building effective predictive models.
  * **Systematic Evaluation:** Comparing the simple test set evaluation with the more comprehensive backtesting approach provided valuable insights. Backtesting gives a much more realistic estimate of a model's performance over time, which is essential for time-dependent data.
  * **Architectural Exploration:** While the primary focus was on a DeepAR-style model, setting up the framework for `NBEATS` and a `gridsearch` demonstrates my commitment to exploring state-of-the-art solutions and systematically optimizing model performance.

> 💡 *Some interactive outputs (e.g., plots, widgets) may not display correctly on GitHub. If so, please view this notebook via [nbviewer.org](https://nbviewer.org) for full rendering.*

-----

## 👤 Author

**Mehran Asgari**

  * **Email:** [imehranasgari@gmail.com](mailto:imehranasgari@gmail.com)
  * **GitHub:** [https://github.com/imehranasgari](https://github.com/imehranasgari)

## 📄 License

This project is licensed under the Apache 2.0 License – see the `LICENSE` file for details.