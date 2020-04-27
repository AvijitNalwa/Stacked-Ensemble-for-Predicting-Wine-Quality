# Wine Quality Prediction Using a Stacked Ensemble Model
A Stacked Ensemble using a Deep Neural Network meta-trainer &amp; Gradient Boosting Machine, Random Forest, Deep Neural Network as base models for predicting the quality of red & white wines.(h20.ai, Pandas, Seaborn)

## Structure

#### Main Files:
1. Wine Quality.py [Data Visualization + Random Forest Modelling + Useful Methods]
2. Wine Quality Ensemble.py [Advanced Version of 1st file, Stacked Ensemble Modelling (Final Model)]
3. Loading Saved Ensembles.py [How to load the saved Stacked Ensemble Model to predict on data]
4. Output .txt files [Model Details, Error Metrics, General Information]
5. red_ensemble4, white_ensemble4 [Stacked Ensemble Saved Model Files]
6. Visualizations .png files [Correlation Matrix Heatmaps, Standardized Variable Distributions, Variable Importances]

## Results
Exact Regression using: 


Normalized RMSE % = (RMSE/(MAX-MIN))*100

Accuracy = 100 - Normalized RMSE %
#### Red Wine Accuracy: 88.2382197087 % 
#### White Wine Accuracy: 90.0720307958 %

## Dataset
#### University of California, Irvine - Machine Learning Repository 
Wine Quality Data Set

Abstract: Two datasets are included, related to red and white vinho verde wine samples, from the north of Portugal. The goal is to model wine quality based on physicochemical tests (Cortez et al., 2009).
http://archive.ics.uci.edu/ml/datasets/Wine+Quality

## Data Visualizations
### Red Wine Variable Correlation Matrix Heatmap
<img src="https://github.com/AvijitNalwa/Wine-Quality/blob/master/Red%20Correlation%20Matrix%20Heatmap.png" width="70%" height="70%">

### White Wine Variable Correlation Matrix Heatmap
<img src="https://github.com/AvijitNalwa/Wine-Quality/blob/master/White%20Correlation%20Matrix%20Heatmap.png" width="70%" height="70%">

### Red Wine Standardized Variables Distribution
<img src="https://github.com/AvijitNalwa/Wine-Quality/blob/master/Red%20Columns%20Normalized%20Distribution%20Plot.png" width="70%" height="70%">

### White Wine Standardized Variables Distribution
<img src="https://github.com/AvijitNalwa/Wine-Quality/blob/master/White%20Columns%20Normalized%20Distribution%20Plot.png" width="70%" height="70%">

## Insights

Interesting results from the numerous runs while training models. 

#### Red Wine Gradient Boosting Machine Variable Importance for Predicting Quality

<img src="https://github.com/AvijitNalwa/Wine-Quality/blob/master/Red%20GBM%20VarImp.png" width="80%" height="80%">



#### Red Wine Random Forest Variable Importance for Predicting Quality

<img src="https://github.com/AvijitNalwa/Wine-Quality/blob/master/Red%20RF%20VarImp.png" width="80%" height="80%">

#### White Wine Gradient Boosting Machine Variable Importance for Predicting Quality

<img src="https://github.com/AvijitNalwa/Wine-Quality/blob/master/White%20GBM%20VarImp.png" width="80%" height="80%">

#### White Wine Random Forest Variable Importance for Predicting Quality

<img src="https://github.com/AvijitNalwa/Wine-Quality/blob/master/White%20RF%20VarImp.png" width="80%" height="80%">


## Resources

h20.ai : http://docs.h2o.ai/h2o/latest-stable/h2o-docs/index.html

pandas: https://pandas.pydata.org/

seaborn: https://seaborn.pydata.org/


