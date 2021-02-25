# Comp432ResearchProject

# README for G37
Our project relies mainly on sklearn and Pytorch

Our code submission contains the following files:

## Files and Folders

./environment.yaml <-- conda environment for our project <br />
./data/ <-- Stock data exmples in csv format  <br /> 
./Linear_Regression/ <-- files to run a linear regression model, includes feature_engineering.py for additional features RSI and ATR<br />

### TCN

./TCN/TCN.py <-- Class to build TCN model, highly customizable by user parameters <br />
./TCN/TCNandOptuna.ipynb <-- Google Colab notebook as proof that Hyper-parameter optimization for TCN was attempted <br />
./TCN/MultivariateTCN.ipynb <-- Python Notebook to play with Multivariate TCN, note that there are many parameters and variable will change model <br />
./TCN/UnivariateTCN.ipynb <-- Python Notebook to play with Univariate TCN, note that parameters and variable will change model <br />
./TCN/Multivariate/ <-- folder contains files for feature engineering and preprocessing for the multivariate model <br />
./TCN/Univariate/ <-- folder contains files for feature engineering and preprocessing for the multivariate model <br />

To run a TCN model, simply run the Python Notebooks


### LSTM
* GPU is not required.
* Training takes ~2 mins.
.LSTM.py is where the LSTM is built including data preprocessing, compiling the model, and testing. Everything is modularized in one file.
.test.ipynb is to run the model and output the graphs and predictions. Just execute cell with run_LSTM() method and it will do everything.

