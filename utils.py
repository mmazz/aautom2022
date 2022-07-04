#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 12:47:43 2017

@author: leandro
"""

import librosa
import numpy as np
import os
from math import ceil
import time
import sys

SAMPLE_RATE = 22050

distanceMatrix = None

def findMusic(directory):
    supportedFileTypes = ["wav", "mp3"] #flac ?
    musicFiles = []

    for file in os.listdir(directory):
        if os.path.isdir(directory + file):
            musicFiles += findMusic(directory + file + "/")
        elif os.path.splitext(file)[1][1:] in supportedFileTypes:
            musicFiles.append( directory + file )
        else:
            if not file.endswith(".asd"):
                print("Skipped:", directory + file)

    return musicFiles

SIZE_STFT = 45100 #esto seguramente se pueda calcular pero bue
def getSTFT(data, superVector = True):
    D = librosa.stft(data)
    D = np.abs(D)

    if superVector:
        return D.reshape( D.shape[0] * D.shape[1] )
    else:
        return D.reshape( (D.shape[1], D.shape[0] ) )

SIZE_ZERO_CROSSING_RATE = 22 #esto seguramente se pueda calcular pero bue
def getZeroCrossingRate(data):
    zc = librosa.feature.zero_crossing_rate(data)
    return zc.reshape( zc.shape[0] * zc.shape[1] )

SIZE_RMSE = 22 #esto seguramente se pueda calcular pero bue
def getRMSE(data):
    rmse = librosa.feature.rmse(data)
    return rmse.reshape( rmse.shape[0] * rmse.shape[1] )

SIZE_MFCC = 440 #esto seguramente se pueda calcular pero bue
def getMFCC(data, superVector = True):
    mfcc = librosa.feature.mfcc(data, sr = SAMPLE_RATE)
    if superVector:
        return mfcc.reshape( mfcc.shape[0] * mfcc.shape[1] )
    else:
        return mfcc.reshape( (mfcc.shape[1], mfcc.shape[0] ) )

def scaleByRow(data):

    scaledData = np.copy(data)

#    if min == None:
#        min = np.min(data)
#
#    if max == None:
#        max = np.max(data)

    #si es una sola matriz
    if len(data.shape) == 2:
        for row in range(0, scaledData.shape[0]):
            min = np.min( scaledData[row,:] )
            max = np.max( scaledData[row,:] )
            scaledData[row, :] = np.divide( ( scaledData[row, :] - min ) , ( max - min) )
    elif len(data.shape) == 3:
        for row in range(0, scaledData.shape[1]):
            min = np.min ( np.min( data[:,row,:], axis = 1 ) )
            max = np.max ( np.max( data[:,row,:], axis = 1 ) )
            scaledData[:, row, :] = np.divide( ( scaledData[:, row, :] - min ) , ( max - min) )


    else:
        scaledData = None


    return scaledData


def unScale(scaledData, min, max):
    return ( scaledData  * ( max - min) ) + min

def getAudioData( audioFiles, superVector = True, features = "stft", audioMaxLength = 3, qtyFilesToProcess = None ):
    count = 0
    countFail = 0
    COUNT_NOTICE = 200
#    COUNT_FAIL = 20

    maxProgress = 0.5

    listAudioData = []

    tic = time.process_time()

    audioFilesDone = []

    sizeAudioRaw = ceil(SAMPLE_RATE * audioMaxLength)

    if qtyFilesToProcess == None:
        qtyFilesToProcess = len(audioFiles)

    for i in range(0, qtyFilesToProcess):
        try:
            file = audioFiles[i]
            sys.stdout.write('.')
            sys.stdout.flush()

            tmpAudioData, tmpSampleRate = librosa.core.load(file, sr = SAMPLE_RATE)

            tmpAudioData.resize(sizeAudioRaw,  refcheck=False)

            featuresData = None

            if features == "mfcc":
                featuresData = getMFCC(tmpAudioData, superVector)
            elif features == "stft":
                featuresData = getSTFT(tmpAudioData)

            listAudioData.append( featuresData )
            audioFilesDone.append(file)

            count += 1


            if count % COUNT_NOTICE == 0:
                sys.stdout.write('\n\r')
                print("[", count, "/", qtyFilesToProcess, "]")
                sys.stdout.flush()

        except Exception as ex:
            countFail += 1
            sys.stdout.write('\n\r')
            print(file, "[FAIL]", ex)
            sys.stdout.flush()



            # if countFail >= COUNT_FAIL:
            #     break

    matrixAudioData = np.array(listAudioData, dtype=np.float32)
#    matrixAudioData = matrixAudioData.squeeze(1)
    audioFiles.clear()
    audioFiles += audioFilesDone

    print("")
    print("Matriz final:", matrixAudioData.shape)

    toc = time.process_time()
    print("time:", toc - tic)
    return matrixAudioData

def saveAudioData( matrixAudioData, filename ):
    np.save(filename, matrixAudioData)

def loadAudioData( filename ):
    return np.load(filename)

def doPCA( matrixAudioData ):
    from sklearn.decomposition import PCA

    tic = time.process_time()

    pca = PCA(n_components=0.98, svd_solver = "full")
    pca.fit(matrixAudioData)
    print("Variance explained:", pca.explained_variance_ratio_.sum())
    matrixAudioDataTransformed = pca.transform(matrixAudioData)

    toc = time.process_time()

    print("shape transformed:", matrixAudioDataTransformed.shape)
    print("time:", toc - tic)
    return matrixAudioDataTransformed

def doDistanceMatrix( matrixAudioDataTransformed ):
    from scipy.spatial import distance as dist
    global distanceMatrix

    distanceFunction = 'cosine' #canberra, cityblock, braycurtis, euclidean

    print("Processing distance matrix...")
    print("Distance function:", distanceFunction)

    tic = time.process_time()
    distanceMatrix = dist.pdist(matrixAudioDataTransformed, distanceFunction)
    toc = time.process_time()
    print("time:", toc - tic)

def doHierachicalClustering( matrixAudioDataTransformed, threshold = 0.992 ):
    from scipy.cluster import hierarchy as h
    from scipy.spatial import distance as dist

    linkageType = 'average' #single, complete, weighted, average

    print("Linkage type:", linkageType)

    if distanceMatrix == None:
        doDistanceMatrix(matrixAudioDataTransformed)

    tic = time.process_time()

    clusters = h.linkage(distanceMatrix, linkageType)
    c,d=h.cophenet(clusters, distanceMatrix) #factor cofonético

    toc = time.process_time()

    print("Cophenet factor:",c)
    print("time:", toc - tic)

    # THRESHOLD = 0.995
    #THRESHOLD = 0.92
    cutTree = h.cut_tree(clusters, height=threshold)

    return cutTree

def doTSNE( matrixAudioDataTransformed, n_components = 2 ):
    from sklearn.manifold import TSNE
    from sklearn.metrics import pairwise_distances
    from scipy.spatial import distance as dist

    tic = time.process_time()

    similarities = pairwise_distances( dist.squareform(distanceMatrix), n_jobs = -1)

    tsne = TSNE(n_components=n_components, metric="precomputed")
    positions = tsne.fit(similarities).embedding_

    toc = time.process_time()

    print("time:", toc - tic)

    return positions

def doDBScan( tsneResult ):
    from sklearn.cluster import DBSCAN

    db = DBSCAN( eps=2, min_samples=3, metric="euclidean" )
    dbFit = db.fit( tsneResult )

    return dbFit.labels_
