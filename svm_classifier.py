import os
import numpy as np
from sklearn import svm
from sklearn import metrics, preprocessing
import matplotlib.pyplot as plt
import time
from datetime import datetime

def main():
    mainStartTime = time.time()
    trainFeaturePath = './features_labels/train/'
    testFeaturePath = './features_labels/test/'
    featureFilename = 'features.csv'
    labelFilename = 'labels.csv'
    encoderFilename = 'encoderClasses.csv'
    print(f'[INFO] ========= TRAINING PHASE ========= ')
    trainFeatures = getFeatures(trainFeaturePath,featureFilename)
    trainEncodedLabels = getLabels(trainFeaturePath,labelFilename)
    svm = trainSVM(trainFeatures,trainEncodedLabels)
    print(f'[INFO] =========== TEST PHASE =========== ')
    testFeatures = getFeatures(testFeaturePath,featureFilename)
    testEncodedLabels = getLabels(testFeaturePath,labelFilename)
    encoderClasses = getEncoderClasses(testFeaturePath,encoderFilename)
    predictedLabels = predictSVM(svm,testFeatures)
    elapsedTime = round(time.time() - mainStartTime,2)
    print(f'[INFO] Code execution time: {elapsedTime}s')
    accuracy = plotConfusionMatrix(encoderClasses,testEncodedLabels,predictedLabels)
    return accuracy

def getFeatures(path,filename):
    features = np.loadtxt(path+filename, delimiter=',')
    return features

def getLabels(path,filename):
    encodedLabels = np.loadtxt(path+filename, delimiter=',',dtype=int)
    return encodedLabels

def getEncoderClasses(path,filename):
    encoderClasses = np.loadtxt(path+filename, delimiter=',',dtype=str)
    return encoderClasses

    
def trainSVM(trainData,trainLabels):
    print('[INFO] Training the SVM model...')
    svm_model = svm.SVC(kernel='linear', C=1, random_state=42)
    startTime = time.time()
    svm_model.fit(trainData, trainLabels)
    elapsedTime = round(time.time() - startTime,2)
    print(f'[INFO] Training done in {elapsedTime}s')
    return svm_model

def predictSVM(svm_model,testData):
    print('[INFO] Predicting...')
    startTime = time.time()
    predictedLabels = svm_model.predict(testData)
    elapsedTime = round(time.time() - startTime,2)
    print(f'[INFO] Predicting done in {elapsedTime}s')
    return predictedLabels

def getCurrentFileNameAndDateTime():
    fileName =  os.path.basename(__file__).split('.')[0] 
    dateTime = datetime.now().strftime('-%d%m%Y-%H%M')
    return fileName+dateTime

def plotConfusionMatrix(encoderClasses,testEncodedLabels,predictedLabels):
    encoder = preprocessing.LabelEncoder()
    encoder.classes_ = encoderClasses
    #Decoding test labels from numerical labels to string labels
    test = encoder.inverse_transform(testEncodedLabels)
    #Decoding predicted labels from numerical labels to string labels
    pred = encoder.inverse_transform(predictedLabels)
    print(f'[INFO] Plotting confusion matrix and accuracy...')
    fig, ax = plt.subplots(figsize=(8, 6))
    metrics.ConfusionMatrixDisplay.from_predictions(test,pred,ax=ax, colorbar=False, cmap=plt.cm.Greens)
    plt.suptitle('Confusion Matrix: '+getCurrentFileNameAndDateTime(),fontsize=18)
    accuracy = metrics.accuracy_score(testEncodedLabels,predictedLabels)*100
    plt.title(f'Accuracy: {accuracy}%',fontsize=18,weight='bold')
    plt.savefig('./results/'+getCurrentFileNameAndDateTime(), dpi=300)  
    print(f'[INFO] Plotting done!')
    print(f'[INFO] Close the figure window to end the program.')
    plt.show(block=False)
    return accuracy

if __name__ == "__main__":
    main()
