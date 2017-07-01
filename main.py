from __future__ import print_function
import librosa
import numpy as np
import os
import glob

# from librosa import librosa.core.zero_crossings
# filename  = librosa.util.example_audio_file()
pathList = [x[0] for x in os.walk("/Users/fbeneditovm/Desktop/Dataset_B")]
auxList = [x[1] for x in os.walk("/Users/fbeneditovm/Desktop/Dataset_B")]
classes = auxList[0]
savedFeatures = []
lines = []

# print(pathList, classes)
file_number = 0
# Browse folders
for i in range(1, len(pathList)):
    files = glob.glob(pathList[i] + "/*.wav")
    # print(files)

    # Browse files
    for file in files:
        # print(files)
        file_number += 1
        if file_number % 10 == 0:
            print("Reading file: " + str(file_number))
        y, sr = librosa.load(file)
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        featureList = []
        featureString = ""
        ##### Features

        # print(librosa.feature.zero_crossing_rate(y))

        zero_crossing = []
        rmse = []
        spectral_bandwidth = []
        spectral_centroid = []
        spectral_rollOff = []

        try:
            zero_crossing = [float(x) for x in librosa.feature.zero_crossing_rate(y)[0]]
        except:
            print("Unable to covert zero_crossing: "+str(file))

        try:
            rmse = [float(x) for x in librosa.feature.rmse(y)[0]]
        except:
            print("Unable to convert rmse "+str(file))

        try:
            spectral_bandwidth = [float(x) for x in librosa.feature.spectral_bandwidth(y)[0]]
        except:
            print("Unable to convert spec_band "+str(file))

        try:
            spectral_centroid = [float(x) for x in librosa.feature.spectral_centroid(y)[0]]
        except:
            print("Unable to convert spec_cent "+str(file))

        try:
            spectral_rollOff = [float(x) for x in librosa.feature.spectral_rolloff(y)[0]]
        except:
            print("Unable to convert spec_roll "+str(file))

        tonnetz = librosa.feature.tonnetz(y)
        spectral_contrast = librosa.feature.spectral_contrast(y)
        mfcc = librosa.feature.mfcc(y)
        #melspec = librosa.feature.melspectrogram(y)
        chroma_cens = librosa.feature.chroma_cens(y)
        chroma_cqt = librosa.feature.chroma_cqt(y)
        chroma_stft = librosa.feature.chroma_stft(y)

        ##### Add Features
        featureList.append(np.mean(rmse))
        featureList.append(np.mean(spectral_centroid))
        featureList.append(np.mean(spectral_bandwidth))
        featureList.append(np.mean(spectral_rollOff))
        featureList.append(np.mean(zero_crossing))

        for t in tonnetz:
            ft = [float(x) for x in t]
            featureList.append(np.mean(ft))
        for c in spectral_contrast:
            ft = [float(x) for x in c]
            featureList.append(np.mean(ft))
        for m in mfcc:
            ft = [float(x) for x in m]
            featureList.append(np.mean(ft))
        # for mel in melspec:
        #    ft = [float(x) for x in mel]
        #    featureList.append(np.mean(ft))
        for ch in chroma_cens:
            ft = [float(x) for x in ch]
            featureList.append(np.mean(ft))
        for cqt in chroma_cqt:
            ft = [float(x) for x in cqt]
            featureList.append(np.mean(ft))
        for stft in chroma_stft:
            ft = [float(x) for x in stft]
            featureList.append(np.mean(ft))


        # Add class at the end
        featureList.append(classes[i - 1])
        if file_number % 10 == 0:
            print("nFeatures: "+str(len(featureList)))

        for feat in featureList:
            featureString += str(feat)+","

        # Add a \n if it is not the final file
        if not(i == (len(pathList)-1) and file == files[-1]):
            featureString += "\n"

        # savedFeatures.append(featureList)
        lines.append(featureString)

f = open('/Users/fbeneditovm/Desktop/Dataset_B/dataset_trainingB.csv', 'w')
f.writelines(lines)
##print(opt.shape)
# print('Saving output to beat_times.csv')
# librosa.output.times_csv('C:\\Users\\gilvan\\Downloads\\TrainingBNormal\\beat_times.txt', opt



