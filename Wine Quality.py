import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import h2o
from h2o.estimators.random_forest import H2ORandomForestEstimator
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


# applying normalization to each column of both datasets
norm_red = red_data.apply(normalize)
print(norm_red.head())
norm_white = white_data.apply(normalize)
print(norm_white.head())


# function to plot all columns on one graph
def plot_cols(data):
    for col in data:
        sn.kdeplot(data[col])
    plt.show()


# plotting all columns on one graph by dataset
plot_cols(norm_red)
plot_cols(norm_white)

# initializing h20 and getting rid of old sessions
h2o.init(max_mem_size="10G")
h2o.remove_all()


def h20_split(data):
    train, test = train_test_split(data, test_size=0.2)
    train_h, test_h = h2o.H2OFrame(train), h2o.H2OFrame(test)
    return train_h, test_h


r_train, r_test = h20_split(norm_red)
w_train, w_test = h20_split(norm_white)


# function returning list of predictors and string of target column name (assumes last column is target)
def get_predictors_target(data):
    predictors = data.columns.values[:-1]
    target = data.columns.values[-1]
    return list(predictors), str(target)


# getting predictors list and target variable
predictors, target = get_predictors_target(norm_red)

# defining random forest model for red wines
r_model = H2ORandomForestEstimator(model_id="rf_red",
                                   ntrees=900,
                                   stopping_rounds=100,
                                   score_each_iteration=True,
                                   max_depth=9000,
                                   seed=1000000)

# supervised training of random forest model for reds
r_model.train(predictors, target, training_frame=r_train, validation_frame=r_test)

# saving trained RF red model
h2o.save_model(model=r_model, path="rf_red", force=True)

# printing key red RF model info
print(r_model)
print(r_model.score_history())

w_model = H2ORandomForestEstimator(model_id="rf_white",
                                   ntrees=900,
                                   stopping_rounds=100,
                                   score_each_iteration=True,
                                   max_depth=9000,
                                   seed=1000000)

# supervised training of random forest model for whites
w_model.train(predictors, target, training_frame=w_train, validation_frame=w_test)

# saving trained RF white model
h2o.save_model(model=w_model, path="rf_white", force=True)

# printing key white RF model info
print(w_model)
print(w_model.score_history())

