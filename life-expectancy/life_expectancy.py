import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from sklearn.compose import ColumnTransformer

dataset = pd.read_csv('life_expectancy.csv')

print(dataset.head())
print(dataset.describe())

dataset.drop(labels=['Country'], axis=1)

labels = dataset.iloc[:, -1]
features = dataset.iloc[:, 0:-1]

features = pd.get_dummies(features)

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.20, random_state=23)

numerical_features = features.select_dtypes(include=['float64', 'int64'])
numerical_columns = numerical_features.columns

ct = ColumnTransformer(transformers=[('numeric', Normalizer(), numerical_columns)], remainder='passthrough')

features_train_scaled = ct.fit_transform(features_train)
features_test_scaled = ct.transform(features_test)

my_model = tf.keras.models.Sequential()
input = tf.keras.layers.InputLayer(input_shape=(features.shape[1], ))
my_model.add(input)
my_model.add(tf.keras.layers.Dense(64, activation='relu'))
my_model.add(tf.keras.layers.Dense(1))

print(my_model.summary())

opt = tf.keras.optimizers.Adam(learning_rate=0.01)
my_model.compile(loss='mse', metrics=['mae'], optimizer=opt)

res_mse, res_mae = my_model.evaluate(features_test, labels_test, verbose=0)

print(res_mse, res_mae)