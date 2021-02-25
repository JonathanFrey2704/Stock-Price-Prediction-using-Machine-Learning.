# Comp432ResearchProject

# README for G37
Our project relies mainly on sklearn and Pytorch

Our code submission contains the following files:

## Files and Folders

./environment.yaml <-- conda environment for our project 
./data/ <-- Stock data exmples in csv format  
./Linear_Regression/ <-- files to run a linear regression model, includes feature_engineering.py for additional features RSI and ATR

### TCN

./TCN/TCN.py <-- Class to build TCN model, highly customizable by user parameters 
./TCN/TCNandOptuna.ipynb <-- Google Colab notebook as proof that Hyper-parameter optimization for TCN was attempted 
./TCN/MultivariateTCN.ipynb <-- Python Notebook to play with Multivariate TCN, note that there are many parameters and variable will change model 
./TCN/UnivariateTCN.ipynb <-- Python Notebook to play with Univariate TCN, note that parameters and variable will change model 
./TCN/Multivariate/ <-- folder contains files for feature engineering and preprocessing for the multivariate model 
./TCN/Univariate/ <-- folder contains files for feature engineering and preprocessing for the multivariate model 

To run a TCN model, simply run the Python Notebooks
* GPU is not required.
* Training takes ~2 mins.


### LSTM
* GPU is not required.
* Training takes ~2 mins.
.LSTM.py is where the LSTM is built including data preprocessing, compiling the model, and testing. Everything is modularized in one file.
.test.ipynb is to run the model and output the graphs and predictions. Just execute cell with run_LSTM() method and it will do everything.

