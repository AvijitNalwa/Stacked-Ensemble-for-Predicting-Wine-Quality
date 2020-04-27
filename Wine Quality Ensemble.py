import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import h2o
from h2o.estimators.random_forest import H2ORandomForestEstimator
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.stackedensemble import H2OStackedEnsembleEstimator
from h2o.estimators.deeplearning import H2ODeepLearningEstimator
from sklearn.model_selection import train_test_split

# loading wine datasets
red_data = pd.read_csv('/Users/Avijit/Downloads/winequality-red.csv', sep=";")
white_data = pd.read_csv('/Users/Avijit/Downloads/winequality-white.csv', sep=";")

# viewing head of both datasets
print(red_data.head())
print(white_data.head())

# viewing ket stats
print(red_data.describe())
print(white_data.describe())

# drop nas if any
red_data.dropna(inplace=True)
white_data.dropna(inplace=True)

# creating correlation matrix for datasets
corr_mx_red = red_data.corr()
corr_mx_white = white_data.corr()

# creating a diverging color palette so that significant correlations are easily visible
cmap = sn.diverging_palette(240, 10, sep=20, as_cmap=True)

# plotting heatmap of correlation matrix for both datasets
sn.heatmap(corr_mx_red, cmap=cmap)
plt.show()
sn.heatmap(corr_mx_white, cmap=cmap)
plt.show()


# function to normalize column data
def normalize(x):
    return (x - x.mean()) / x.std()


for col in red_data.columns:
    if col != 'quality':
        red_data[col] = normalize(red_data[col])
        white_data[col] = normalize(white_data[col])


# applying normalization to each column of both datasets
norm_red = red_data #.apply(normalize)
print(norm_red.head())
norm_white = white_data #.apply(normalize)
print(norm_white.head())


# function to plot all columns on one graph
def plot_cols(data):
    for col in data:
        sn.kdeplot(data[col])
    plt.show()


# plotting all columns on one graph by dataset
plot_cols(red_data)
plot_cols(white_data)

print(red_data.head())
print(white_data.head())

# initializing h20 and getting rid of old sessions
h2o.init(max_mem_size="10G")
h2o.remove_all()


def h20_split(data):
    train, test = train_test_split(data, test_size=0.2)
    train_h, test_h = h2o.H2OFrame(train), h2o.H2OFrame(test)
    return train_h, test_h


r_train, r_test = h20_split(norm_red)
w_train, w_test = h20_split(norm_white)

rh_data = h2o.H2OFrame(red_data)
wh_data = h2o.H2OFrame(white_data)


# function returning list of predictors and string of target column name (assumes last column is target)
def get_predictors_target(data):
    predictors = data.columns.values[:-1]
    target = data.columns.values[-1]
    return list(predictors), str(target)


# getting predictors list and target variable
predictors, target = get_predictors_target(norm_red)

r_gbm = H2OGradientBoostingEstimator(ntrees=200,
                                     learn_rate=0.001,
                                     nfolds=6,
                                     keep_cross_validation_predictions=True,
                                     # stopping_rounds=100,
                                     # stopping_tolerance=1e-4,
                                     max_depth=200,
                                     seed=1)

r_gbm.train(predictors, target, training_frame=rh_data)

h2o.save_model(model=r_gbm, path="gbm_red2", force=True)

# print(r_gbm)

# defining random forest model for red wines
r_rf = H2ORandomForestEstimator(model_id="rf_red2",
                                nfolds=6,
                                ntrees=200,
                                keep_cross_validation_predictions=True,
                                # stopping_rounds=100,
                                score_each_iteration=True,
                                max_depth=1000,
                                seed=1)

# supervised training of random forest model for reds
r_rf.train(predictors, target, training_frame=rh_data)

r_deep = H2ODeepLearningEstimator(hidden=[200, 100, 100, 10, 10, 10], epochs=1000, rate=0.001, nfolds=6,
                                  keep_cross_validation_predictions=True, seed=1)
r_deep.train(predictors, target, training_frame=rh_data)

r_stack = H2OStackedEnsembleEstimator(metalearner_algorithm="deeplearning",
                                      metalearner_params={"hidden": [200, 100, 100, 10, 10, 10], "epochs": 100,
                                                          "nesterov_accelerated_gradient": True},
                                      model_id="ensemble4",
                                      training_frame=rh_data,
                                      metalearner_nfolds=6,
                                      base_models=[r_gbm, r_rf, r_deep])

r_stack.train(predictors, target, training_frame=rh_data)
print(r_stack)

h2o.save_model(model=r_stack, path="stack_red3", force=True)

# saving trained RF red model
h2o.save_model(model=r_rf, path="rf_red2", force=True)

# printing key red RF model info
print(r_rf)
print(r_rf.score_history())

w_gbm = H2OGradientBoostingEstimator(ntrees=200,
                                     learn_rate=0.001,
                                     nfolds=6,
                                     keep_cross_validation_predictions=True,
                                     # stopping_rounds=100,
                                     # stopping_tolerance=1e-4,
                                     max_depth=200,
                                     seed=1)

w_gbm.train(predictors, target, training_frame=wh_data)

h2o.save_model(model=w_gbm, path="gbm_white2", force=True)

print(w_gbm)

w_rf = H2ORandomForestEstimator(model_id="rf_white2",
                                ntrees=200,
                                nfolds=6,
                                keep_cross_validation_predictions=True,
                                # stopping_rounds=100,
                                score_each_iteration=True,
                                max_depth=1000,
                                seed=1)

# supervised training of random forest model for whites
w_rf.train(predictors, target, training_frame=wh_data)

w_deep = H2ODeepLearningEstimator(hidden=[200, 100, 100, 10, 10, 10], epochs=1000, rate=0.001, nfolds=6,
                                  keep_cross_validation_predictions=True, seed=1)
w_deep.train(predictors, target, training_frame=wh_data)


w_stack = H2OStackedEnsembleEstimator(model_id="ensemble4",
                                      metalearner_algorithm="deeplearning",
                                      metalearner_params={"hidden": [200, 100, 100, 10, 10, 10], "epochs": 100,
                                                          "nesterov_accelerated_gradient": True},
                                      training_frame=wh_data, metalearner_nfolds=3,
                                      base_models=[w_gbm, w_rf, w_deep],
                                      seed=1)

w_stack.train(predictors, target, training_frame=wh_data)
print(w_stack)

h2o.save_model(model=w_stack, path="stack_white3", force=True)

# saving trained RF white model
h2o.save_model(model=w_rf, path="rf_white2", force=True)

# printing key white RF model info
print(w_rf)
print(w_rf.score_history())
