# Baseline Model for EEG Classification using MiniROCKET and Ridge Classifier

# Accuracy is low due to the small dataset size and the nature of the data. Plus we're running the training on the run '0' for the first subject only.

import numpy as np 
from sktime.transformations.panel.rocket import MiniRocket
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import mne
from moabb.datasets import BNCI2014_001 # Key for BCI IV Dataset 2a

# loading the data

dataset = BNCI2014_001()
dataDictionary = dataset.get_data(subjects=[1])

# first training run 

raw = dataDictionary[1]['0train']['0']
print(f"EEG Data Shape: {raw.get_data().shape}")

# extracting events and labels 

events, eventDictionary = mne.events_from_annotations(raw)
print(f"Events: {events}")
print(f"Event Dictionary: {eventDictionary}")

# creating epochs

epochs = mne.Epochs(raw, events, event_id=eventDictionary, tmin=0, tmax=2, baseline=None, preload=True)

# X, y values

X = epochs.get_data()
y = epochs.events[:, -1]

print(f"X Shape: {X.shape}")
print(f"y Shape: {y.shape}")

# minirocket pipeline

XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.2, random_state=42)

transformer = MiniRocket()
transformer.fit(XTrain)
TransformedXTrain = transformer.transform(XTrain)
TransformedXTest = transformer.transform(XTest)

print(f"Transformed X Train Shape: {TransformedXTrain.shape}")

# ridge classifier baseline model

clf = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), store_cv_values=True)
clf.fit(TransformedXTrain, yTrain)

# predictions and evaluations 

yPred = clf.predict(TransformedXTest)
accuracy = accuracy_score(yTest, yPred)
report = classification_report(yTest, yPred, output_dict=True)

print(f"Accuracy: {accuracy}")
print(f"Classification Report: {report}")