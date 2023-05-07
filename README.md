# Importing libraries
import librosa
import parselmouth
from parselmouth.praat import call
import csv
import sounddevice as sd
import soundfile as sf
import pandas as pd
import numpy as np
import pickle
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Loading the dataset
df = pd.read_csv ("parkinsons.csv")
df.head ()

# Splitting the features and the target
X = df.drop ( ["name", "status"], axis=1)
y = df ["status"]

# Splitting the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split (X, y, test_size=0.2, random_state=42)

# Creating and fitting the SVM model
svm = SVC (kernel="rbf", gamma="auto")
svm.fit (X_train, y_train)

# Making predictions on the test set
y_pred = svm.predict (X_test)

# Evaluating the model performance
acc = accuracy_score (y_test, y_pred)
cm = confusion_matrix (y_test, y_pred)
print ("Accuracy: ", acc)
print ("Confusion matrix: \n", cm)

# Saving the model
pickle.dump (svm, open ("svm_model.pkl", "wb"))

# Loading the model
svm = pickle.load (open ("svm_model.pkl", "rb"))

# Calculating the voice measures for a new voice recording (using the same code as before)

# Setting parameters
samplerate = 44100 # Sample rate
duration = 10 # Duration in seconds
filename = "voice.wav" # File name

# # Recording audio
# print ("Start recording...")
# data = sd.rec (int (samplerate * duration), samplerate=samplerate, channels=1)
# sd.wait () # Wait until recording is finished
# print ("Stop recording...")

# # Saving audio
# sf.write (filename, data, samplerate)

# Loading voice sample
y, sr = librosa.load (filename)
y_trimmed, index = librosa.effects.trim (y) # trim the silent parts
sf.write ("voice.wav", y_trimmed, sr) # save the trimmed recording

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
# Calculating MDVP:Shimmer (%),MDVP:Shimmer (dB),Shimmer:APQ3,
# Shimmer:APQ5,Shimmer:APQ11,Shimmer:DDA - Several measures of variation in amplitude
shimmer_local = parselmouth.praat.call ([sound, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
shimmer_local_dB = parselmouth.praat.call ([sound, pointProcess], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
shimmer_apq3 = parselmouth.praat.call ([sound, pointProcess], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
shimmer_apq5 = parselmouth.praat.call ([sound, pointProcess], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
shimmer_apq11 = parselmouth.praat.call ([sound, pointProcess], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
shimmer_dda = parselmouth.praat.call ([sound, pointProcess], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

# Calculating NHR,HNR - Measures of ratio of noise to tonal components in the voice
noise_harmonics_ratio = parselmouth.praat.call (sound,"To Harmonicity (cc)", 0.01 ,75 ,0.1 ,1)
harmonics_noise_ratio = parselmouth.praat.call (noise_harmonics_ratio,"Get mean",0 ,0)

# Calculating RPDE,D2 - Two nonlinear dynamical complexity measures
sound = parselmouth.Sound ("voice.wav")
pointProcess = parselmouth.praat.call (sound,"To PointProcess (periodic, cc)",75 ,600)
sound2 = parselmouth.praat.call (pointProcess,"PointProcess: To Sound (pulse train)",44100 ,0.7 ,0.05 ,30)
degree_of_voice_breaks = parselmouth.praat.call (sound2,"To DegreeOfVoiceBreaks",0 ,1 ,75 ,600 ,1 ,1 ,1)
d2 = parselmouth.praat.call (degree_of_voice_breaks,"Get standard deviation",0 ,0)

# Calculating DFA - Signal fractal scaling exponent
detrended_fluctuation_analysis = parselmouth.praat.call (sound,"To DetrendedFluctuationAnalysis",1 ,4 ,4 ,4 ,4 ,4)

# Calculating PPE - A nonlinear measure of fundamental frequency variation 
pitch_perturbation_entropy = parselmouth.praat.call (sound,"To PitchPerturbationEntropy",75 ,600 ,40)

# Calculating F1,F2,F3,F4 - Formant frequencies for the vowel /a/
formants = sound.to_formant_burg ()
f1_mean = call(formants,"Get mean",1 ,0 ,0 )
f2_mean = call(formants,"Get mean",2 ,0 ,0 )
f3_mean = call(formants,"Get mean",3 ,0 ,0 )
f4_mean = call(formants,"Get mean",4 ,0 ,0 )

# Creating a feature vector for the new voice sample
new_features = [f0_mean,f0_max,f0_min,jitter_percent,jitter_abs,jitter_rap,jitter_ppq,jitter_ddp,
                shimmer_local_dB,
                shimmer_apq3,
                shimmer_apq5,
                shimmer_apq11,
                shimmer_dda,
                harmonics_noise_ratio,
                d2,
                detrended_fluctuation_analysis,
                pitch_perturbation_entropy,
                f1_mean,f2_mean,f3_mean,f4_mean]

# Reshaping the feature vector to match the input shape of the model
new_features = np.array(new_features).reshape(1,-1)

# Making prediction on the new voice sample using the SVM model
new_pred = svm.predict(new_features)

# Printing the prediction result
if new_pred == 1:
    print("The voice sample belongs to a person with Parkinson's disease.")
else:
    print("The voice sample belongs to a person without Parkinson's disease.")
