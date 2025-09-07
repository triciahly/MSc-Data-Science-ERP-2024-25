# MSc-Data-Science-ERP-2024-25

# Project Overview
This study examines the use of time series forecasting models to predict stock movements in the IT sector of the US stock market over a 24-year period (2000-2024), specifically the top 50 firms ranked by market capitalisation.

This repository evaluates a broad range of models for daily excess return forecasting, including both traditional machine learning models and time-series foundation models (TSFMs):
- **Linear Models**: OLS, Ridge, Lasso, ElasticNet, Generalised Linear Model, Principal Component Regression 
- **Non-Linear Models**: Random Forest, Gradient Boosted Regression Trees 
- **Neural Networks**: NN1–NN5 
- **Time-Series Foundation Models (TSFMs)**: Chronos (T5/Bolt), TimesFM, Moirai (uni2ts)

# Project Structure
The main folders/files of the project are shown
```
|
|----- 📁 01_Data_Preprocessing_&_EDA
|       |----- 📓 Data_Preprocessing_&_EDA.ipynb
|
|----- 📁 02_Benchmark_Models
|       |----- 📓 Benchmark_Models_&_Portfolio_Results.ipynb
|
|----- 📁 03_Time_Series_Foundation_Models
|       |----- 📓 Chronos_&_Portfolio_Results.ipynb
|       |----- 📓 Moirai_&_Portfolio_Results.ipynb
|       |----- 📓 TimesFM_&_Portfolio_Results.ipynb
|
|----- 📁 04_Diebold_Marino_Test_Results
|       |----- 📓 Diebold_Marino_Test_Results.ipynb
|       |----- 📁📗 DM_Test_Data.csv
|
|----- 📁 05_Cumulative_Log_Returns_Visulisations
|       |----- 📓 Cumulative_Log_Returns.ipynb
|       |----- 📁📗 Cumulative_Log_Returns_Data.csv
|
|----- 📰 README.md
|
|----- 🛒 requirements.txt
|
|----- 🛒 requirements_benchmark.txt
|
|----- 🛒 requirements_chronos.txt
|
|----- 🛒 requirements_moirai.txt
|
|----- 🛒 requirements_timesfm.txt
|
|----- 👩‍⚖️ LICENSE
|
|----- 🤷‍♀️ .gitignore
|

```
* 📁 **01_Data_Preprocessing_&_EDA** - Script for cleaning, preprocessing, and exploratory data analysis of financial datasets.
* 📁 **02_Benchmark_Models** - Script for running traditional machine learning models (linear, non-linear, neural networks).
* 📁 **03_Time_Series_Foundation_Models** - Scripts for running advanced TSFMs, including Chronos, Moirai (Uni2TS), and TimesFM models.
* 📁 **04_Diebold_Marino_Test_Results** - Statistical test outputs for pairwise comparison between benchmark models and TSFMs.
* 📁 **05_Cumulative_Log_Returns_Visulisations** - Cumulative log returns of top performing benchmark models and TSFMs.
* 🛒 **requirements.txt** - Dependencies for preprocessing, EDA, results and visualisations.
* 🛒 **requirements_benchmark.txt** - Dependencies for benchmark models (linear, non-linear, neural networks).
* 🛒 **requirements_chronos.txt** - Dependencies for Chronos models
* 🛒 **requirements_moirai.txt** - Dependencies for Moirai (Uni2ts) models
* 🛒 **requirements_timesfm.txt** - Dependencies for TimesFM models

# Methodology
### Data Context
- **Stock Pool**: U.S. Information Technology sector
- **Period**: January 2000 - December 2024
- **Sample**: Top 50 stocks by market capitalisation
- **Features**: Rolling windows (5, 21, 252, 512) days of past excess returns
- **Estimation Period**: 2000-01-01 to 2015-12-31 (Training dataset)
- **Out-of-Sample Period**: 2016–01-01 to 2024-12-31 (Testing dataset)
- **Target Variable**: Daily excess returns

### Evaluation Framework
1. Statistical Evaluation
   - Zero-based R-squared, Mean Square Error, Mean Absolute Error
   - Directional Accuracy (Total, Upward, Downward)
2. Economic Evaluation
   - Portfolio construction based on model forecasts
   - Long-only, Short-only, Long-Short strategies
   - Equal-weighted and Value-weighted portfolios (With and Without Transaction Cost)
   - Primary Risk-adjusted metrics: Sharpe ratio, Maximum Drawdown

# Getting Started

### 1. Clone the repository
```code
# Clone the repository
git clone https://github.com/triciahly/MSc-Data-Science-ERP-2024-25.git

# Navigate into the repository folder
cd MSc-Data-Science-ERP-2024-25
```

### 2. Create and activate a Python Virtual Environment
```code
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```
⚠️ Note: Optional but recommended to create a dedicated virtual environment for each model type to ensure isolation and prevent dependency conflicts.

### 3. Install required packages
Install the relevant requirements.txt file according to your intended use case:

a. For general use: Data Preprocessing, EDA, Results & Visualisations
```code
pip install -r requirements.txt
```

b. For Benchmark models
```code
pip install -r requirements_benchmark.txt
```

c. For TSFM Chronos models
```code
pip install -r requirements_chronos.txt
```

d. For TSFM Moirai (Uni2ts) models
```code
pip install -r requirements_moirai.txt
```

e. For TSFM TimesFM models
```code
pip install -r requirements_timesfm.txt
```

### 4. TSFMs Setup
Download the relevant TSFMs from their official repositories according to your intended use case:

a. Chronos-T5 (Amazon Science) - https://github.com/amazon-science/chronos-forecasting
```code
git clone https://github.com/amazon-science/chronos-forecasting
```

b. TimesFM (Google Research) - https://github.com/google-research/timesfm
```code
git clone https://github.com/google-research/timesfm
```
      
c. Moirai/Uni2ts (Salesforce AI Research) - https://github.com/SalesforceAIResearch/uni2ts
```code
git clone https://github.com/SalesforceAIResearch/uni2ts
```

### 5. Data Acquisition
⚠️ Note: Non-subscribers must obtain permission from CRSP through legitimate WRDS subscription prior to the use of any CRSP data or information in any materials, research, or products. Redistribution of CRSP data is prohibited.

**WRDS Data Access**:
1. [WRDS](https://wrds-www.wharton.upenn.edu/) (Wharton Research Data Services)  Access: Required for data retrieval. (requires institutional subscription)
2. With Direct SQL queries, obtain the following:
   - Date Range: January 1, 2000 - December 31, 2024
   - SIC Codes: 3570-3579, 3600-3674, 7370-7379, 4810-4813 (Programming, Software, etc.)
   - Information variables: PERMNO, PERMCO, DATE, SHROUT, PRC, RET, SHRCD, EXCHCD, SICCD, NCUSIP, COMNAM, TICKER

**Fama-French Factor Data**:
1. [WRDS](https://wrds-www.wharton.upenn.edu/) (Wharton Research Data Services)  Access: Required for data retrieval. (requires institutional subscription)
2. Query the Fama-French factors daily dataset
   - Specify the Columns to Retrieve: daily risk-free rate, DATE
   - Date Range: January 1, 2000 - December 31, 2015 (Training), January 1, 2016 - December 31, 2024 (Testing)
  
**S&P 500 Data**: 
1. Use Yahoo Finance (yfinance) to download the S&P 500 index data for date ranging 2016-01-01 to 2024-12-31.
2. Select the 'Adjusted Close' price, falling back to the 'Close' price if necessary.

# Usage
1. **Data Preprocessing and Exploratory Data Analysis (EDA)**
   - Run the *Data_Preprocessing_&_EDA.ipynb* file to generate cleaned datasets and exploratory plots.
     
2. **Benchmark Models & TSFMs**
   - Run the relevant *.ipynb* files for each model to perform forecasting of daily excess returns and portfolio construction.

      - ⚠️ Note for ALL models: If the .ipynb files do not execute properly in your local environment or Python IDE (e.g., VS Code, PyCharm), this may be due to personal system or environment settings on your device. In such cases, you can run the .ipynb files directly in Google Colab or Jupyter Notebook.

      - ⚠️ Note for Moirai: Running this file will trigger a restart prompt — please accept it to complete the installation and apply all changes correctly.

      - ⚠️ Note for TimesFM: After running the setup, you'll need to manually configure Python 3.11 from the following commands: When prompted, type **2** and press **ENTER** to select Python 3.11 to apply changes.
```code
   !sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 
   !sudo update-alternatives --config python3
```

3. **Diebold Marino Test Results**
   - Run the *Diebold_Marino_Test_Results.ipynb* file to display the pairwise comparison results between benchmark models and TSFM
     
4. **Cumulative Log Returns Visualisation**
   - Run the *Cumulative_Log_Returns.ipynb* to display cumulative log return plots of top performing benchmark models and TSFMs

# Additional Information
### Minimum Requirements
- Python: 3.9+ (3.10 recommended for optimal compatibility with time series foundation models)
- RAM: 16GB+ (32GB+ recommended for large foundation models like Chronos and Moirai)
- Storage: 10GB+ free space for datasets, model checkpoints, and stock data
- GPU: Optional but recommended for faster model training (CUDA)

### License 
This project is licensed under the MIT License - see the LICENSE.md file for details

This research was conducted as part of the requirements for a degree at the University of Manchester. The findings and conclusions are solely those of the author and do not necessarily reflect the views of the University.
