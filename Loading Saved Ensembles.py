import h2o
import pandas as pd

h2o.init(max_mem_size="10G")
h2o.remove_all()

red_data = pd.read_csv('/Users/Avijit/Downloads/winequality-red.csv', sep=";")
white_data = pd.read_csv('/Users/Avijit/Downloads/winequality-white.csv', sep=";")

def normalize(x):
    return (x - x.mean()) / x.std()


for col in red_data.columns:
    if col != 'quality':
        red_data[col] = normalize(red_data[col])
        white_data[col] = normalize(white_data[col])

print(red_data.head())
print(white_data.head())

rh_data = h2o.H2OFrame(red_data)
wh_data = h2o.H2OFrame(white_data)

red_ensemble = h2o.load_model('/Users/Avijit/Desktop/Implementations/Wine Quality Prediction/stack_red3/ensemble4')
white_ensemble = h2o.load_model('/Users/Avijit/Desktop/Implementations/Wine Quality Prediction/stack_white3/ensemble4')

print(red_ensemble)
print(white_ensemble)

r_preds = red_ensemble.predict(rh_data)
w_preds = white_ensemble.predict(wh_data)
print(r_preds)
print(w_preds)







