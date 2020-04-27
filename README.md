# Wine Quality Prediction Using a Stacked Ensemble Model
A Stacked Ensemble using a Deep Neural Network meta-trainer &amp; Gradient Boosting Machine, Random Forest, Deep Neural Network as base models for predicting the quality of red & white wines.(h20.ai, Pandas, Seaborn)

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

## Data Visualization
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

Variable Importances:
variable              relative_importance    scaled_importance    percentage
--------------------  ---------------------  -------------------  ------------
alcohol               218907                 1                    0.274382
sulphates             114727                 0.524089             0.1438
volatile acidity      103534                 0.472958             0.129771
total sulfur dioxide  63151.6                0.288485             0.0791551
pH                    54651.4                0.249655             0.0685008
free sulfur dioxide   44358.7                0.202637             0.0555999
fixed acidity         44191                  0.201871             0.0553896
citric acid           41245.8                0.188417             0.0516981
density               39054.3                0.178406             0.0489513
chlorides             37168                  0.169789             0.0465869
residual sugar        36831.6                0.168252             0.0461653

#### Red Wine Random Forest Variable Importance for Predicting Quality

Variable Importances:
variable              relative_importance    scaled_importance    percentage
--------------------  ---------------------  -------------------  ------------
alcohol               208382                 1                    0.214607
sulphates             140795                 0.67566              0.145001
volatile acidity      125056                 0.600128             0.128792
density               78805.4                0.378178             0.0811596
citric acid           74434.8                0.357204             0.0766584
total sulfur dioxide  72093.9                0.34597              0.0742476
fixed acidity         58441.3                0.280453             0.0601871
pH                    56932.6                0.273213             0.0586333
chlorides             56815                  0.272648             0.0585122
free sulfur dioxide   50113.2                0.240487             0.0516102
residual sugar        49124.7                0.235744             0.0505922

#### White Wine Gradient Boosting Machine Variable Importance for Predicting Quality

Variable Importances:
variable              relative_importance    scaled_importance    percentage
--------------------  ---------------------  -------------------  ------------
alcohol               605686                 1                    0.246652
volatile acidity      308943                 0.510071             0.12581
free sulfur dioxide   283837                 0.468621             0.115587
pH                    186544                 0.307988             0.0759661
total sulfur dioxide  167275                 0.276174             0.068119
fixed acidity         163079                 0.269247             0.0664105
density               160890                 0.265633             0.0655191
sulphates             157653                 0.260288             0.0642006
residual sugar        155995                 0.257551             0.0635256
citric acid           140124                 0.231348             0.0570626
chlorides             125598                 0.207365             0.0511472

#### White Wine Random Forest Variable Importance for Predicting Quality

Variable Importances:
variable              relative_importance    scaled_importance    percentage
--------------------  ---------------------  -------------------  ------------
alcohol               559947                 1                    0.190521
density               320678                 0.572693             0.10911
volatile acidity      310310                 0.554178             0.105583
free sulfur dioxide   296229                 0.52903              0.100792
chlorides             237977                 0.425                0.0809716
total sulfur dioxide  232302                 0.414865             0.0790406
pH                    209604                 0.374328             0.0713174
citric acid           202676                 0.361956             0.0689604
fixed acidity         195650                 0.349408             0.0665697
residual sugar        189165                 0.337828             0.0643634
sulphates             184486                 0.329471             0.0627712

## Resources

h20.ai : http://docs.h2o.ai/h2o/latest-stable/h2o-docs/index.html

pandas: https://pandas.pydata.org/

seaborn: https://seaborn.pydata.org/


