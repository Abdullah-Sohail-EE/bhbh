# Importing libraries
import librosa
import parselmouth
from parselmouth.praat import call
import csv
import sounddevice as sd
import soundfile as sf
import pandas as pd
import numpy as np
import nolds
import pickle
import scipy
from scipy import stats
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV # Added GridSearchCV for hyperparameter tuning
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score 
# Loading the dataset
df = pd.read_csv ("pk1.csv")
df.head ()

# Splitting the features and the target
X = df.drop ( ["name", "status"], axis=1)
y = df ["status"]

# Splitting the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split (X, y, test_size=0.2, random_state=42)

# Creating and fitting the SVM model with hyperparameter tuning using GridSearchCV
svm = SVC () # Removed the kernel and gamma parameters to let GridSearchCV find the best ones
parameters = {'kernel':('linear', 'rbf'), 'C':[0.1, 1, 10], 'gamma':('scale', 'auto')} # Defined a parameter grid to search over
clf = GridSearchCV(svm, parameters) # Created a GridSearchCV object with the svm model and the parameter grid
clf.fit (X_train, y_train) # Fitted the GridSearchCV object on the training data

# Making predictions on the test set using the best estimator from GridSearchCV
y_pred = clf.best_estimator_.predict (X_test)

# Evaluating the model performance
acc = accuracy_score (y_test, y_pred)
cm = confusion_matrix (y_test, y_pred)
f1 = f1_score (y_test, y_pred)
sensitivity = recall_score (y_test, y_pred)
print ("Accuracy: ", acc)
print ("Confusion matrix: \n", cm)
print ("F1 score: ", f1)
print ("Sensitivity: ", sensitivity)

# Saving the model using the best estimator from GridSearchCV
pickle.dump (clf.best_estimator_, open ("svm_model.pkl", "wb"))

# Loading the model
svm = pickle.load (open ("svm_model.pkl", "rb"))

# Calculating the voice measures for a new voice recording (using the same code as before)
# Setting parameters
samplerate = 44100 # Sample rate
duration = 10 # Duration in seconds
filename = "voice.wav" # File name

# Recording audio
print ("Start recording...")
data = sd.rec (int (samplerate * duration), samplerate=samplerate, channels=1)
sd.wait () # Wait until recording is finished
print ("Stop recording...")

# Saving audio
sf.write (filename, data, samplerate)

# Loading voice sample
y, sr = librosa.load (filename)
y_trimmed, index = librosa.effects.trim (y) # trim the silent parts
sf.write ("voice.wav", y_trimmed, sr) # save the trimmed recording

# Converting voice sample to numpy array
y = np.asarray(y_trimmed) # convert to numpy array

# Calculating MDVP:Fo (Hz) - Average vocal fundamental frequency
f0, voiced_flag, voiced_probs = librosa.pyin (y, fmin=librosa.note_to_hz ('C2'), fmax=librosa.note_to_hz ('C7'))
f0_mean = np.nanmean (f0)

# Calculating MDVP:Fhi (Hz) - Maximum vocal fundamental frequency
f0_max = np.nanmax (f0)

# Calculating MDVP:Flo (Hz) - Minimum vocal fundamental frequency
f0_min = np.nanmin (f0)

# Calculating MDVP:Jitter (%),MDVP:Jitter (Abs),MDVP:RAP,MDVP:PPQ,Jitter:DDP - Several measures of variation in fundamental frequency
sound = parselmouth.Sound ("voice.wav")
pitch = sound.to_pitch ()
pointProcess = parselmouth.praat.call ([sound, pitch], "To PointProcess (cc)")
jitter_percent = parselmouth.praat.call (pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
jitter_abs = parselmouth.praat.call (pointProcess, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
jitter_rap = parselmouth.praat.call (pointProcess, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
jitter_ppq = parselmouth.praat.call (pointProcess, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
jitter_ddp = parselmouth.praat.call (pointProcess, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3)


# Calculating MDVP:Shimmer (%),MDVP:Shimmer (dB),Shimmer:APQ3,Shimmer:APQ5,MDVP:APQ,Sshimmer:DDA - Several measures of variation in amplitude
# Creating a point process object from the sound and pitch objects
pointProcess = parselmouth.praat.call([sound, pitch], "To PointProcess (cc)")
# Calculating local shimmer
shimmer_local = parselmouth.praat.call([sound, pointProcess], "Get shimmer (local)", 0.0 ,0.0 ,0 ,0 ,0.0001 ,1.3) 
# Calculating local shimmer in dB
shimmer_local_db = parselmouth.praat.call([sound, pointProcess], "Get shimmer (local_dB)", 0.0 ,0.0 ,0 ,0 ,0.0001 ,1.3) 
# Calculating apq3 shimmer
shimmer_apq3 = parselmouth.praat.call([sound, pointProcess], "Get shimmer (apq3)", 0.0 ,0.0 ,0 ,0 ,0.0001 ,1.3) 
# Calculating apq5 shimmer
shimmer_apq5 = parselmouth.praat.call([sound, pointProcess], "Get shimmer (apq5)", 0.0 ,0.0 ,0 ,0 ,0.0001 ,1.3) 
# Calculating apq11 shimmer
shimmer_apq11 = parselmouth.praat.call([sound, pointProcess], "Get shimmer (apq11)", 0.0 ,0.0 ,0 ,0 ,0.0001 ,1.3) 
# Calculating dda shimmer
shimmer_dda = parselmouth.praat.call([sound, pointProcess], "Get shimmer (dda)", 0.0 ,0.0 ,0 ,0 ,0.0001 ,1.3)


# Calculating NHR,HNR - Two measures of ratio of noise to tonal components in the voice
harmonicity = parselmouth.praat.call (sound.to_harmonicity (), "Get mean", 0 ,0)
nhr = parselmouth.praat.call (sound.to_harmonicity (), "Get standard deviation", 0 ,0)

# Extracting a part of the sound object with 131072 samples
sound = parselmouth.praat.call (sound, "Extract part", 0, 5.9, "rectangular", 1, "no")

# Converting sound object to spectrum object using FFT
spectrum = parselmouth.praat.call (sound, "To Spectrum")

# Printing the spectrum object
print(spectrum)

# Printing the sound object
print(sound)

# Calculating RPDE,D2 - Two nonlinear dynamical complexity measures
rpde = parselmouth.praat.call (spectrum, "Get standard deviation",1)
# spectrum = parselmouth.praat.call (sound, "To Spectrum")
# spectrum_values = spectrum.values # Getting the values of the spectrum object as a numpy array
# spectrum_values = spectrum_values.flatten() # Flattening the 2D array to a 1D array
# d2 = nolds.corr_dim(spectrum_values, emb_dim=10) # Calculating the correlation dimension using nolds optimized function
#Calculating DFA - Signal fractal scaling exponent
dfa = nolds.dfa(y)

# Calculating spread1 and spread2 - Two spectral measures of the variation of fundamental frequency
spread1 = stats.skew(y)
spread2 = stats.kurtosis(y)

# Calculating PPE - A nonlinear measure of fundamental frequency variation 
ppe = np.std(y) / np.sqrt(len(y))

# Creating a feature vector from the voice measures
features = [f0_mean,f0_max,f0_min,jitter_percent,jitter_abs,jitter_rap,jitter_ppq,jitter_ddp,
            shimmer_local,shimmer_local_db,shimmer_apq3,shimmer_apq5,shimmer_apq11,shimmer_dda,
            nhr,harmonicity,rpde,dfa,spread1,spread2,ppe]

# Reshaping the feature vector to a 2D array
features = np.array(features).reshape(1,-1)

# Creating a data frame with the voice measures
df = pd.DataFrame(features, columns=["f0_mean","f0_max","f0_min","jitter_percent","jitter_abs","jitter_rap","jitter_ppq","jitter_ddp",
            "shimmer_local","shimmer_local_db","shimmer_apq3","shimmer_apq5","shimmer_apq11","shimmer_dda",
            "nhr","harmonicity","rpde","dfa","spread1","spread2","ppe"])

# Saving the data frame as a csv file
df.to_csv("voice_measures.csv", index=False)

# Making a prediction using the loaded model
prediction = svm.predict(features)

# Printing the prediction
if prediction == 0:
    print("The voice sample belongs to a healthy person.")
else:
    print("The voice sample belongs to a person with Parkinson's disease.")
