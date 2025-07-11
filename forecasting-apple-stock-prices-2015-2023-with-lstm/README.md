**Forecasting Apple Stock Prices (2015–2023) with LSTM**

This project builds an LSTM-based time series forecasting model to predict Apple's (AAPL) daily closing price using historical price and volume data along with technical indicators. It includes data collection, feature engineering, preprocessing, model training, evaluation, and visualization.

---

## 🚀 Features

- **Data Acquisition**: Automatically downloads historical price data (Open, High, Low, Close, Volume) using `yfinance`.
- **Technical Indicators**: Calculates 10-day Moving Average and 14-day RSI to enrich features.
- **Data Visualization**: Plots price evolution and trading volume.
- **Sequence Generation**: Converts time series into 60-day sequences suitable for LSTM input.
- **LSTM Model**: Stacked LSTM with dropout for next-day closing price forecasting.
- **Invert Scaling to Original Price Space**: Reconstruct USD prices from normalized predictions for interpretability.
- **Generate Predictions & Visualizations**: Visualize actual vs. predicted Close across train + val + test in one subplot.
- **Evaluation**: Reports MAE and RMSE on train, validation, and test sets.


---

## 📦 Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/hassan-obaya/Projects-/forecasting-apple-stock-prices-2015-2023-with-lstm.git
   cd forecasting-apple-stock-prices-2015-2023-with-lstm
   ```
2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate    # macOS/Linux
   venv\Scripts\activate     # Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🛠 Usage

Run interactively in a Jupyter notebook:

```bash
jupyter notebook
# Open and run `notebook.ipynb`
```

### Parameters to tweak

- `SEQ_LEN` (sequence length, default 60)
- `EPOCHS`, `BATCH_SIZE` for training
- Technical indicator windows (e.g., change `MA_10` to other periods)

---

## 📂 Project Structure
forecasting-apple-stock-prices-2015-2023-with-lstm/

```
├── README.md                  
├── notebook.ipynb             
├── figures/                  
│    └── prediction_plot.png     
├── requirements.txt            

---

## 📈 Results

After training for 50 epochs, the model typically achieves:

```
Performance Metrics (USD):
Train      MAE:     1.20 | RMSE:     1.98
Val        MAE:     3.81 | RMSE:     5.13
Test       MAE:     5.75 | RMSE:     6.76
```

## 🙌 Contributing

Contributions are welcome! Feel free to open issues, suggest new indicators or models, or improve documentation.


