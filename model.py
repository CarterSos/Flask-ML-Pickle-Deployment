import pandas as pd
import joblib
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pickle as pkl
from sklearn.model_selection import train_test_split


df_classes = pd.read_csv('class.csv')
#print(df_classes.shape)
#df_classes

df = pd.read_csv('zoo.csv')
#print(df.shape)
#df.head()

# y = df.class_type
# X = df.drop(columns=['class_type','animal_name'])

for col in df:
  if not pd.api.types.is_numeric_dtype(df[col]) and col != 'class_type' and col != 'animal_name':
    df = pd.get_dummies(df, columns=[col], drop_first=True)

y = df.class_type
X = df.drop(columns=['class_type','animal_name'])
# X = pd.get_dummies(X, drop_first=True) 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=4) # state = 4 for reptile example
# model = RandomForestClassifier(random_state=4).fit(X_train, y_train)
model = DecisionTreeClassifier(max_depth=5).fit(X_train, y_train)

print(f'Accuracy:\t{model.score(X_test, y_test)}')
print(y_test.value_counts() / y_test.shape[0])

# model = DecisionTreeClassifier(max_depth=5) 
# model.fit(X, y)

# # Example input features
# # hair, feathers, eggs, milk, airborne, aquatic, predator, toothed, backbone, breathes, venomous, fins, legs, tail, domestic, catsize
# input_features = [0,1,1,0,1,0,0,0,0,1,0,0,2,0,0,0]  # Replace with actual values
# input_features = [0,0,1,0,1,0,0,0,0,0,1,0,6,0,0,0]
# # Reshape input_features to match the expected format
# input_features = np.array(input_features).reshape(1, -1)

# # Predict class
# predicted_class = model.predict(input_features)
# predictionList = ['mammal','bird','reptile','fish','amphibian','bug','invertebrate']
# # Print predicted class
# print(predictionList[predicted_class[0] - 1])

joblib.dump(model, "model.joblib")
pkl.dump(model, open("model.pkl", "wb"))