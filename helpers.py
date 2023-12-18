######################################################################## imports
#######################################################################

import numpy as np
import scipy as sp
import sounddevice as sd
import re
import math
import time
import random
import matplotlib.pyplot as plt
from operator import itemgetter

#################################################################### subroutines
####################################################################

## miscelleaneous
##

def average_adiag(x):
    """Average antidiagonal elements of a 2d array
    Parameters:
    -----------
    x : np.array
        2d numpy array of size

    Return:
    -------
    x1d : np.array
        1d numpy array representing averaged antediangonal elements of x

    """
    x1d = [np.mean(x[::-1, :].diagonal(i)) for i in
           range(-x.shape[0] + 1, x.shape[1])]
    x1d = np.array(x1d)
    x1d = x1d.reshape(x1d.shape[0], 1)
    return np.array(x1d)

## audio subroutines
##

def recordAudioToVector(sampleRate, duration):
    '''
    Record audio from computer microphone and store as single channel data in a 
    numpy array\n
    Inputs:\n
    \t  sampleRate: sample rate to use for recording, measured in Hz
    \t\t  (number of samples per second)\n
    \t  duration: length to record for, in seconds\n
    Outputs:\n
    \t  audioArray: (m x 1) numpy vector, where m = sampleRate * duration\n
    '''
    
    frames = int(duration * sampleRate)
    
    print("Recording started!")
    audioArray = sd.rec(frames=frames, samplerate=sampleRate, channels=1)
    
    print("Recording...")
    # wait for recording to finish
    sd.wait()
    
    print("Recording finished!")

    return audioArray

def audioVectorToWavFile(audioVector: np.array, sampleRate, fileName: str):
    '''
    Convert a data vector of an audio recording to a .wav file\n
    Inputs:\n
    \t  audioVector: (m x 1) numpy vector to convert\n
    \t  sampleRate: sample rate of audioVector, measuring in Hz
    \t\t  (number of samples per second)\n
    \t  fileName: name of file to create WITHOUT the extension
    Outputs:\n
    \t None
    '''

    newFileName = fileName + ".wav"
    sp.io.wavfile.write(newFileName, sampleRate, audioVector)

    return

def wavFileToAudioVector(fileName: str):
    '''
    Convert a mono-channel .wav audio file to a numpy array\n
    Inputs:\n
    \t  fileName: name of file to convert WITH extension\n
    Outputs:\n
    \t  (sampleRate, audioVector) where:\n
    \t\t  sampleRate: detected sampleRate of .wav file, measured in Hz 
    \t\t\t  (number of samples per second)\n
    \t\t  audioVector: (m x 1) numpy array
    '''

    sampleRate, audioVector = sp.io.wavfile.read(fileName)
    # resize audioVector from (m, ) to (m, 1)
    audioVector = audioVector.reshape(audioVector.shape[0], 1)

    return (sampleRate, audioVector)

def findSpeechPause(audioVector: np.array, sampleFrequency, windowDuration):
    '''
    Given a data vector representing an audio file of someone talking, find a 
    find a portion of that matrix where a speech pause occurs\n
    Inputs:\n
    \t  audioVector: (m x 1) vector of someone speaking\n
    \t  sampleFrequency: sample frequency of audioVector, measured in Hz
    \t\t (number of samples / s)\n
    \t  windowDuration: length of rolling window [s]
    Outputs:\n
        pauseVector: (l x 1) vector containing data of a speech pause
    '''

    samplesInWindow = math.floor(sampleFrequency * windowDuration)
    duration = audioVector.shape[0] / sampleFrequency
    numWindows = math.ceil(duration / windowDuration)

    windowStartIndex = 0
    minAverage = 1e10
    minAverageIndex = -1
    for i in range(0, numWindows):
        if i == numWindows - 1:
            windowVector = audioVector[windowStartIndex:]
        else:
            windowVector = audioVector[windowStartIndex : windowStartIndex +\
                                        samplesInWindow]
            
        windowAverage = np.average(np.abs(windowVector))
        if windowAverage < minAverage:
            minAverage = windowAverage
            minAverageIndex = windowStartIndex
        windowStartIndex += samplesInWindow

    print(minAverage)
    print(minAverageIndex)
    #print(audioVector[minAverageIndex:minAverageIndex + samplesInWindow])

    return audioVector[minAverageIndex:minAverageIndex + samplesInWindow]

def addWhiteNoise(audioVector: np.array, mu, eta, debug=False):
    '''
    Add Gaussian white noise with mean mu and standard deviation eta\n
    Inputs:\n
    \t  audioVector: (m x 1) audio vector to add the white noise to\n
    \t  mu: desired mean of Gaussian white noise\n
    \t  eta: desired standard deviation of Gaussian white noise, which will 
    \t    have variance (eta)^2\n
    \t  debug: set to True to output debugging information, default False
    Outputs:\n
        whiteAudioVector: (m x 1) audioVector with added white noise\n
    '''

    numSamples = audioVector.shape[0]
    whiteNoise = np.random.normal(mu, eta, size=numSamples)
    whiteNoise = whiteNoise.reshape(whiteNoise.shape[0], 1)

    whiteAudioVector = audioVector + whiteNoise

    # debugging
    if debug:
        samples = np.arange(0, numSamples, 1)
        plt.figure("1")
        plt.plot(samples, audioVector)
        plt.plot(samples, whiteNoise)
        plt.legend(["Original", "White noise"])
        plt.figure("2")
        plt.plot(samples, whiteAudioVector)
        plt.legend(["Combined"])
        plt.show()

    return whiteAudioVector

def removeWhiteNoiseSVD(audioVector: np.array, sampleRate, eta,\
                        gainMethod: str, extractMethod: str, tolMethod: str,\
                            safetyFactor, windowMethod: str,\
                                windowDuration=0.03, windowOverlap=0,\
                                    debug=False):
    '''
    Remove white noise from an audio vector\n
    Inputs:\n
    \t  audioVector: (m x 1) audio vector\n
    \t  sampleRate: sample rate (in Hz, samples / second) of audioVector\n
    \t  eta: (estimated) value of the standard deviation of the white 
    \t    noise, where (eta)^2 is the (estimated) variance of the white noise\n
    \t  gainMethod: method to use to form gain matrix: either "LS" for least 
    \t    squares, "MLS" for modified least squares, "MV" for minimum variance, 
    \t    or "TDC" for time-domain constraint\n
    \t  extractMethod: method to use to extract the processed audio vector from 
    \t    the Hankel matrix, "ROW" for row extraction or "AD" for averaging 
    \t    the antidiagonals\n
    \t  tolMethod: tolerance to use for determing the numerical rank, k, 
    \t    of the Hankel matrix, "SQRT(M)*ETA" to use sqrt(m) * eta, "M*ETA^2" 
    \t    to use m * eta^2\n
    \t  safetyFactor: safety factor to use to ensure that that the numerical 
    \t    rank k of the Hankel matrix is an underestimate
    \t  windowMethod: method used to process audio, either "BLOCKWISE" or 
    \t    "SLIDING"\n
    \t  windowDuration: duration of each window in seconds, default is 
    \t   0.03 s = 30 ms\n
    \t  windowOverlap: if the "BLOCKWISE" windowMethod is specified, determines 
    \t    how much overlap is used between successive windows, measured in 
    \t    seconds, default is 0 s
    \t  debug: set True to output debugging information, default False
    Outputs:\n
    \t  cleanAudioVector: (m x 1) audioVector, now with the white noise removed
    '''

    # size of audioVector
    (mAV, nAV) = audioVector.shape
    
    ### NOTE: brackets [] represent units, e.g. [s] represents units of 
    #   seconds, used to make sure calculations are being done right 
    #   (dimensional analysis type beat)
    
    ## - calculations to split audioVector into windows with a duration of 
    #    windowDuration ms

    # [samples]
    totalSamples = mAV

    # [s] = ( [samples] ) / ( [samples] / [s] )
    audioVectorDuration = totalSamples / sampleRate

    # window calculations for blockwise windows with no overlap between windows
    if windowMethod == "BLOCKWISE" and windowOverlap == 0:
        # - include extra windows by rounding up since there might not be an 
        #   integer amount of windows with length windowDuration inside an 
        #   audioVectorDuration signal
        # - [windows] = [s] * ( [windows] / [s] )
        estNumWindows = math.ceil(audioVectorDuration / windowDuration)

        # - round down since we've included extra windows
        # - ( [samples] / [window] ) = [samples] / [windows]
        samplesPerWindow = math.floor(totalSamples / estNumWindows)

        # - how many samples are we missing from rounding?
        # - [samples] = [windows] * ( [samples] / [window] )
        numCoveredSamples = estNumWindows * samplesPerWindow
        # [samples] = [samples] - [samples]
        numMissedSamples = totalSamples - numCoveredSamples
        # add extra window if we are missing samples
        if numMissedSamples != 0:
            numWindows = estNumWindows + 1
        else:
            numWindows = estNumWindows
    else:
        ## TODO: implement window calculations when overlap is used between 
        #  blockwise windows as well as when sliding windows are used
        print("Haven't implemented sliding windows or windows with overlap \
              yet!")
        
    # debugging
    if debug:
        print("")
        print("DEBUG OUTPUT: window calculations")
        print("")

        print("Total number of samples in audioVector:", str(totalSamples),\
            "samples")
        print("Sample rate of audioVector:", str(sampleRate),\
              "samples per second")
        print("Calculated duration of audioVector:", str(audioVectorDuration),\
            "seconds")
        print("")
        
        print("Desired window overlap:", str(windowOverlap), "seconds")
        print("Desired duration of each window:", str(windowDuration), "seconds")
        print("")
        
        print("Estimated number of windows needed:", str(estNumWindows), "windows")
        print("Calculated number of samples per window :", str(samplesPerWindow),\
            "samples / window")
        print("")
        
        print("Number of samples covered using estimate of", str(estNumWindows),\
            "windows and", str(samplesPerWindow), "samples per window:",\
                str(numCoveredSamples), "/",  str(totalSamples), "samples")
        print("Number of samples not covered using the above estimates:",\
            str(numMissedSamples), "samples")
        print("Calculated samples to use in additional last window:",\
            str(numMissedSamples))
        print("")
        if numMissedSamples != 0:
            print("SUMMARY: using", str(estNumWindows), "+ 1 windows, with",\
                str(samplesPerWindow), "samples per window for all windows",\
                    "except for the last one, which will use",\
                        str(numMissedSamples), "samples")
            print(str(estNumWindows), "*", str(samplesPerWindow), "+",\
                str(numMissedSamples), "=",\
                    str(estNumWindows * samplesPerWindow + numMissedSamples))
        else:
            print("SUMMARY: using", str(numWindows), "windows, with",\
                str(samplesPerWindow), "samples per window for all windows")
            print(str(numWindows), "*", str(samplesPerWindow), "=",\
                str(numWindows * samplesPerWindow))
        print("_______________________________________________________________")
        print("")
        
    ## process audioVector window by window
    #

    # processed audio vector goes here
    cleanAudioVector = np.zeros([mAV, 1])
    
    # iterate over all windows
    windowStartIndex = 0
    windowEndIndex = windowStartIndex + samplesPerWindow
    for i in range(0, numWindows):
        # - create audio vector for current window
        # - make sure last window has the extra samples (if there are any)
        if i == numWindows - 1:
            windowAudioVector = audioVector[windowStartIndex : ]
        else:
            windowAudioVector = audioVector[windowStartIndex : windowEndIndex]

        (mWin, nWin) = windowAudioVector.shape
        
        # create Hankel matrix H and find its singular values
        H = sp.linalg.hankel(windowAudioVector)
        svListH = np.linalg.svd(H, compute_uv=False)

        # - determine tolerance, the value that the singular values of H must be 
        #   greater than, based on tolMethod
        if tolMethod == "SQRT(M)*ETA":
            tol = np.sqrt(mAV) * eta
        elif tolMethod == "M*ETA^2":
            tol = mAV * (eta ** 2)
        else:
            print("ERROR! UNKNOWN TOLERANCE METHOD SPECIFIED IN",\
                  "removeWhiteNoiseSVD()")
            return
        tol *= safetyFactor
        
        # - calculate k, the numerical rank of H, which is the number of 
        #   singular values of H that are greater than tolerance
        k = sum(1 for ii in svListH if ii > tol)
        # enforce a bare minimum numerical rank of 1
        if k == 0:
            k += 1

        # create the rank k SVD approximation of H
        kCalc, U, Sigma, QT, Hhat, Pk, SigmaK, QTk, duration =\
            SVDrankKApproximation(H, k=k)
        
        # create the gain matrix phi based on gainMethod
        mEta2SigmaNeg2 = mAV * (eta ** 2) * np.linalg.matrix_power(SigmaK, -2)
        if gainMethod == "LS":
            phi = np.eye(k)
        elif gainMethod == "MLS":
            phi = sp.linalg.sqrtm(np.eye(k) - mEta2SigmaNeg2)
        elif gainMethod == "MV":
            phi = np.eye(k) - mEta2SigmaNeg2
        elif gainMethod == "TDC":
            # - lagrange parameter, can be changed, value of 0 leads to LS 
            #   method while value of 1 leads to MV method
            LAMBDA = 2
            phi = (np.eye(k) - mEta2SigmaNeg2) *\
                               (np.eye(k) - ((1 - LAMBDA) * mEta2SigmaNeg2))
        else:
            print("ERROR! UNKNOWN GAIN METHOD SPECIFIED IN",\
                  "removeWhiteNoiseSVD()")
            return
            
        # (re)compute svdHhat
        Hhat = Pk @ phi @ SigmaK @ QTk
        (mHhat, nHhat) = Hhat.shape

        # extract clean signal s hat from Hhat depending on extractMethod
        if extractMethod == "ROW":
            # arbitrary row of Hhat
            shat = np.transpose(Hhat[0])
        elif extractMethod == "AD":
            # average along the antidiagonals of Hhat
            shat = np.zeros(mHhat)
            for iii in range(mHhat):
                # - TODO: fix runtime warning from taking the mean of an empty
                #   slice
                shat[iii] = np.mean(np.diag(np.fliplr(Hhat),\
                                            math.ceil(nHhat / 2) - iii))
        else:
            print("ERROR! UNKNOWN EXTRACT METHOD SPECIFIED IN",\
                  "removeWhiteNoiseSVD()")
            return
        shat = shat.reshape(shat.shape[0], 1)
        (ms, ns) = shat.shape

        # - append shat to the complete processed audio vector
        # - make sure to account for the last window
        if i == numWindows - 1:
            cleanAudioVector[windowStartIndex :] = shat
        else:
            cleanAudioVector[windowStartIndex : windowEndIndex] = shat
        
        # more debugging
        if debug:
            print("DEBUG OUTPUT: numerical ranks of H matrices")
            print("")

            print("i (window number):", str(i), "/", str(numWindows - 1))
            print("windowAudioVector shape:", str(windowAudioVector.shape))
            print("")
            
            r = np.linalg.matrix_rank(H)
            print("Dimensions of H:", str(H.shape))
            print("Rank of H: r =", str(r))
            print("Calculated numerical rank of H: k =", str(k))

            print("Dimensions of shat:", str(shat.shape))
            print("")
            print("")

        
        windowStartIndex = windowEndIndex
        windowEndIndex += samplesPerWindow

    # even more debugging
    if debug:
        print("_______________________________________________________________")
        print("")
        print("DEBUG OUTPUT: cleanAudioVector")
        print("")

        print("Shape of cleanAudioVector")
        print(cleanAudioVector.shape)
        print("")

        samples = np.arange(0, totalSamples)
        plt.figure()
        plt.plot(samples, audioVector)
        plt.legend(["Signal with white noise"])

        plt.figure()
        plt.plot(samples, cleanAudioVector)
        plt.legend(["Processed signal"])

        plt.show()

    return cleanAudioVector

## formating subroutines
##

def truncateNumber(num, precision):
    '''
    Truncates a number to the number of decimal places specified and
    returns the number as a string. If truncation cannot be done, the original
    number is returned as a string.\n
    Inputs:\n
    \t  num: number to be truncated\n
    \t  precision: number of decimal places desired\n
    Outputs:\n
    \t  strNum: truncated version of number, now a string\n
    '''

    # convert num to string
    strNum = str(num)

    # number has decimal point and is in scientific notation
    scientific = "e" in strNum and "." in strNum
    # number has decimal point, not in scientific notation
    decimal = "e" not in strNum and "." in strNum

    # find indices of e's and .'s, then truncate
    if scientific:
        pointIndex = strNum.index(".")
        eIndex = strNum.index("e")
        decimals = len(strNum[pointIndex:eIndex])
        # make sure we can actually truncate
        if precision > decimals:
            return strNum
        return strNum[0 : pointIndex + precision + 1] + strNum[eIndex :]
    elif decimal:
        pointIndex = strNum.index(".")
        decimals = len(strNum[pointIndex:]) - 1
        # make sure we can actually truncate
        if precision > decimals:
            return strNum
        return strNum[0 : pointIndex + precision + 1]
    else:
        # nothing to truncate
        return strNum

def prettyPrintFactorization(A: np.array, B: np.array, C: np.array,\
                             D: np.array, precision, Aname="A", Bname="B",\
                                Cname="C", Dname="D"):
    '''
    Pretty print the matrix factorization A = B @ C @ D with included sizes and 
    optional matrix names. \n
    
    Print format is below (doesn't display well in intellisense):\n
    [A_11 A_12 ... A_1n]   
    [A_21 A_22 ... A_2n] = 
    [ .    .    .   .  ]
    [A_m1 A_m2 ... A_mn]
                   Aname
                     m x n

    [B_11 B_12 ... B_1n]   [C_11 C_12 ... C_1n]   [D_11 D_12 ... D_1n]
    [B_21 B_22 ... B_2n] @ [C_21 C_22 ... C_2n] @ [D_21 D_22 ... D_2n]
    [ .    .    .   .  ]   [ .    .    .   .  ]   [ .    .    .   .  ]
    [B_n1 B_n2 ... B_nn]   [C_n1 C_n2 ... C_nn]   [D_n1 D_n2 ... D_nn]
                   Bname                  Cname                  Dname
                     m x n                  m x n                  m x n
    
    Inputs:\n
    \t  A, B, C, D: matrices of any size\n
    \t  precision: number of decimal places to display for matrix elements\n
    \t  Aname, Bname, Cname, Dname: (optional) names for the printed matrices\n
    Outputs:\n
    \t  None
    '''

    # get sizes, (m x n), of A, B, C, D
    Am, An = A.shape[0], A.shape[1]
    Bm, Bn = B.shape[0], B.shape[1]
    Cm, Cn = C.shape[0], C.shape[1]
    Dm, Dn = D.shape[0], D.shape[1]

    # figure out which row to print operators "=" and "@"'s
    maxRowSize = max(Am, Bm, Cm, Dm)
    operatorRow = math.floor(maxRowSize / 2)

    # find element with max length of each matrix
    maxElementLengthA = max([len(truncateNumber(element, precision))\
                             for row in A for element in row])
    maxElementLengthB = max([len(truncateNumber(element, precision))\
                             for row in B for element in row])
    maxElementLengthC = max([len(truncateNumber(element, precision))\
                             for row in C for element in row])
    maxElementLengthD = max([len(truncateNumber(element, precision))\
                             for row in D for element in row])
    
    # surprise tool that will help us later
    firstRow = ""

    # - pretty printing time!
    # - iterate over each row value, from [0 to maxRowSize - 1]
    for rowIndex in range(0, maxRowSize):
        ## rows of A matrix
        ##
        
        # make sure A actually has elements at row rowIndex
        if rowIndex >= Am:
            aRow = " "
            aRow = aRow * (An * (maxElementLengthA + 1) + 3)
            if rowIndex == operatorRow:
                aRow += "@ "
            else:
                aRow += "  "
            print(aRow, end="")
        else:
            aRow = ""
            # open bracket of A
            aRow += "["
            # iterate over columns of A
            for colIndex in range(0, An):
                # print element, adding spaces as necessary
                Aelement = truncateNumber(A[rowIndex][colIndex], precision)
                while len(Aelement) < maxElementLengthA:
                    Aelement = " " + Aelement
                aRow += Aelement
                # print spaces between elements, sometimes
                if colIndex != An - 1:
                    aRow += " "
            # close bracket of A
            aRow += "] "
            # add equals sign if in correct row
            if rowIndex == operatorRow:
                aRow += "= "
            else:
                aRow += "  "
            print(aRow, end="")

        ## rows of B matrix
        ##

        # make sure B actually has elements at row rowIndex
        if rowIndex >= Bm:
            bRow = " "
            bRow = bRow * (Bn * (maxElementLengthB + 1) + 3)
            if rowIndex == operatorRow:
                bRow += "@ "
            else:
                bRow += "  "
            print(bRow, end="")
        else:
            bRow = ""
            # open bracket of B
            bRow += "["
            # iterate over columns of B
            for colIndex in range(0, Bn):
                # print element, adding spaces as necessary
                Belement = truncateNumber(B[rowIndex][colIndex], precision)
                while len(Belement) < maxElementLengthB:
                    Belement = " " + Belement
                bRow += Belement
                # print spaces between elements, sometimes
                if colIndex != Bn - 1:
                    bRow += " "
            # close bracket of B
            bRow += "] "
            # add multiplication sign if in correct row
            if rowIndex == operatorRow:
                bRow += "@ "
            else:
                bRow += "  "
            print(bRow, end="")

        ## rows of C matrix
        ## 

        # make sure C actually has elements at row rowIndex
        if rowIndex >= Cm:
            cRow = " "
            cRow = cRow * (Cn * (maxElementLengthC + 1) + 2)
            if rowIndex == operatorRow:
                cRow += "@ "
            else:
                cRow += "  "
            print(cRow, end="")
        else:
            cRow = ""
            # open bracket of C
            cRow += "["
            # iterate over columns of C
            for colIndex in range(0, Cn):
                # make sure B actually has elements at row rowIndex
                if rowIndex >= Cm:
                    continue
                # print element, adding spaces as necessary
                Celement = truncateNumber(C[rowIndex][colIndex], precision)
                while len(Celement) < maxElementLengthC:
                    Celement = " " + Celement
                cRow += Celement
                # print spaces between elements, sometimes
                if colIndex != Cn - 1:
                    cRow += " "
            # close bracket of C
            cRow += "] "
            # add multiplication sign if in correct row
            if rowIndex == operatorRow:
                cRow += "@ "
            else:
                cRow += "  "
            print(cRow, end="")

        ## rows of D matrix
        ##
        
        # make sure D actually has elements at row rowIndex
        if rowIndex >= Dm:
            dRow = " "
            dRow = dRow * (Dn * (maxElementLengthD + 1) + 3)
            print(dRow)
        else:
            dRow = ""
            # open bracket of D
            dRow += "["
            # iterate over columns of D
            for colIndex in range(0, Dn):
                # make sure B actually has elements at row rowIndex
                if rowIndex >= Dm:
                    continue
                # print element, adding spaces as necessary
                Delement = truncateNumber(D[rowIndex][colIndex], precision)
                while len(Delement) < maxElementLengthD:
                    Delement = " " + Delement
                dRow += Delement
                # print spaces between elements, sometimes
                if colIndex != Dn - 1:
                    dRow += " "
            # close bracket of D
            dRow += "] "
            print(dRow)

        # save first row for later use
        if rowIndex == 0:
            firstRow = aRow + bRow + cRow + dRow
        
    # find indices of string where "[" character is from first row
    brackIndices = [m.start() for m in re.finditer("\]", firstRow)]

    ## matrix names
    ##

    # build name row
    matrixNameRow = ""
    while len(matrixNameRow) <= (brackIndices[0] - len(Aname)):
        matrixNameRow += " "
    matrixNameRow += Aname
    while len(matrixNameRow) <= (brackIndices[1] - len(Bname)):
        matrixNameRow += " "
    matrixNameRow += Bname
    while len(matrixNameRow) <= (brackIndices[2] - len(Cname)):
        matrixNameRow += " "
    matrixNameRow += Cname
    while len(matrixNameRow) <= (brackIndices[3] - len(Dname)):
        matrixNameRow += " "
    matrixNameRow += Dname
    print(matrixNameRow)

    ## matrix sizes
    ##

    # build size row
    matrixSizeRow = ""
    while len(matrixSizeRow) < (brackIndices[0] - len(str(Am)) - 1):
        matrixSizeRow += " "
    matrixSizeRow += str(Am) + " x " + str(An)
    while len(matrixSizeRow) < (brackIndices[1] - len(str(Bm)) - 1):
        matrixSizeRow += " "
    matrixSizeRow += str(Bm) + " x " + str(Bn)
    while len(matrixSizeRow) < (brackIndices[2] - len(str(Cm)) - 1):
        matrixSizeRow += " "
    matrixSizeRow += str(Cm) + " x " + str(Cn)
    while len(matrixSizeRow) < (brackIndices[3] - len(str(Dm)) - 1):
        matrixSizeRow += " "
    matrixSizeRow += str(Dm) + " x " + str(Dn)
    print(matrixSizeRow)

    return

## matrix generation subroutines
##

def rand_sing(dim, rank, tol=1.e-6):
    # randomly generate rank singular values that are larger than the tol
    
    large_sing = np.zeros(rank)
    small_sing = np.zeros(dim - rank)
    
    for i in range(rank):
        large_sing[i] = random.uniform(10, 420)
        
    for j in range(dim - rank):
        small_sing[j] = random.uniform(0, tol)
        
    diag = np.append(large_sing, small_sing)

    np.random.shuffle(diag)

    
    diag_matrix = np.diag(diag)
    
    return diag_matrix

def generate_matrix(m,n,r):

    U = sp.stats.ortho_group.rvs(m)
    V = sp.stats.ortho_group.rvs(n)

    if(m < n):
        Sigma = rand_sing(m,r)
        block = np.zeros([m, n-m])
        Sigma = np.block([Sigma, block])
    elif(m > n):
        Sigma = rand_sing(n,r)
        block = np.zeros([m-n,n])
        Sigma = np.block([[Sigma],[block]])
    else:
        Sigma = rand_sing(m,r)

    return U@Sigma@V

## SVD subroutines
##

def computeSVD(A: np.array):
    '''
    Compute the Singular Value Decomposition of matrix A.\n
    Inputs:\n
    \t  A: (m x n) matrix with rank r > 0\n
    Outputs:\n
    \t  (P, singularValues, Q^T, duration) where:\n
    \t\t    P: (m x r) matrix with orthonormal columns\n
    \t\t    singularValues: (1 x r) matrix of A's unique singular values\n
    \t\t    QT: (r x n) matrix with orthonormal columns\n
    \t\t    duration: time required to compute SVD\n
    '''

    # compute P, Q^T, and the singular values of A
    start = time.time()
    (P, singularValues, QT) = np.linalg.svd(A, full_matrices=False)
    end = time.time()
    duration = end - start

    return (P, singularValues, QT, duration)

def SVDrankKApproximation(A: np.array, k=0):
    '''
    Approximates matrix A of rank r with matrix A_k of rank 1 <= k < r using 
    the SVD\n
    Inputs:\n
    \t  A: (m x n) matrix to approximate\n
    \t  k: (optional, default 0) rank of desired approximation matrix, default 
    \t\t  value of 0 will choose a value for k based on A's singular values\n
    Outputs:\n
    \t  (kCalc, P, Sigma, QT, Ak, Pk, Sigmak, QTk, duration) where:\n
    \t\t  kCalc: rank used for k\n
    \t\t  P: (m x r) matrix with orthonormal columns\n
    \t\t  Sigma: (r x r) diagonal matrix of A's unique singular values\n
    \t\t  QT: (r x n) matrix with orthonormal columns\n
    \t\t  Ak: (m x n) approximation of A\n
    \t\t  Pk: (m x k) matrix containing the leftmost k columns of P\n
    \t\t  Sigmak: (k x k) diagonal matrix of the k largest singular values of 
    \t\t\t  A\n
    \t\t  QTk: (k x n) matrix containing the topmost k rows of QT\n
    \t\t  duration: time taken to create SVD of A\n
    '''

    # compute SVD of A
    P, singularValues, QT, duration = computeSVD(A)
    # create Sigma = diag(sigma_1, ..., sigma_r)
    Sigma = np.diag(singularValues)

    # find r, the rank of A
    r = np.linalg.matrix_rank(A)

    # make sure k is valid
    if k > r:
        print("ERROR in SVDrankKApproximation: k not less than or equal to", \
              "rank of A")
        return(A, None, None, None)
    elif k == 0:
        kCalc = calculateK(singularValues)
    else:
        kCalc = k
    # compute approximation and output it
    Pk = P[:, 0:kCalc]
    Sigmak = Sigma[0:kCalc, 0:kCalc]
    QTk = QT[0:kCalc, :]
    Ak = Pk @ Sigmak @ QTk

    return(kCalc, P, Sigma, QT, Ak, Pk, Sigmak, QTk, duration)

def calculateK(singularValues: np.array):
    '''
    Given a list of singular values of some matrix A, finds a value of k to use 
    for the approximating matrix A_k by examining the changes in order between 
    consecutive singular values\n
    Inputs:\n
    \t  singularValues: (1 x r) matrix of singular values, arranged in 
    \t\t  decreasing order\n
    Outputs:\n
    \t  k: rank to use for Ak\n
    '''

    # find max difference of order
    orderDiffList = [(singularValues[i], np.log10(singularValues[i]) -\
                     np.log10(singularValues[i + 1]), singularValues[i + 1])\
                        for i in range(0, len(singularValues) - 1)]
    maxOrderDiff = max(orderDiffList, key=itemgetter(1))
    # index determines k value
    k = orderDiffList.index(maxOrderDiff) + 1
    
    return k

## QR subroutines
##

def permutedQR1(A, k=0, tol=1.e-3):

    start = time.time()
    Q = A.copy()
    forcedRank = True
    [m,n] = Q.shape
    if(k == 0):
        k = n
        forcedRank = False

    p = np.array(range(n))

    c = np.zeros(n)

    for j in range(n):
        v = Q[:,j]
        c[j] = np.matmul(np.transpose(v),v)

    r = 0

    for i in range(k):
        
        max = i
        maxNorm = c[i]
        for j in range(i + 1, n):
            norm = c[j]
            if(norm > maxNorm):
                max = j
                maxNorm = norm
        if(abs(maxNorm) < tol and not forcedRank):
            break
        r += 1
        if(max != i):
            temp = Q[:,i].copy()
            Q[:,i] = Q[:,max]
            Q[:,max] = temp

            p[i], p[max] = p[max], p[i]
            c[i], c[max] = c[max], c[i]
        
        ui = Q[:,i]
        ui = ui / np.linalg.norm(ui)
        Q[:,i] = ui

        for k in range(i + 1, n):
            xk = Q[:,k]
            inner = np.inner(ui,xk)
            xk = xk - (inner * ui)
            Q[:,k] = xk
            v = Q[i:,k]
            c[k] = np.matmul(np.transpose(v),v)

    P = np.zeros([n,n])
    for i in range(n):
        P[p[i],i] = 1

    R = np.matmul(np.matmul(np.transpose(Q),A),P)

    end = time.time()
    duration = end - start
    
    return [Q,R,P,r,duration]

def QRGivens(A,k=0,tol=1.e-3):


    start = time.time()
    R = A.copy()
    [m,n] = R.shape
    c = np.zeros(n)
    Q = np.eye(m)
    p = np.array(range(n))

    


    for j in range(n):
        v = R[:,j]
        c[j] = np.matmul(np.transpose(v),v)

    for l in range(n):
        
        max = l
        maxNorm = c[l]
        for j in range(l + 1, n):
            norm = c[j]
            if(norm > maxNorm):
                max = j
                maxNorm = norm
        if(max != l):
            temp = R[:,l].copy()
            R[:,l] = R[:,max]
            R[:,max] = temp
            c[l], c[max] = c[max], c[l]
            p[l], p[max] = p[max], p[l]
        

        for j in range(l+1,n):
            c[j] = c[j] - (R[l,j]**2)

        for j in range(l):
            G = buildGivens(l,j,R[j,j],R[l,j],m)
            R = np.matmul(G,R)
            Q = np.matmul(Q,np.transpose(G),Q)

    P = np.zeros([n,n])
    for i in range(n):
        P[p[i],i] = 1

    end = time.time()

    duration = end - start

    return [Q,R,P,duration]

def buildGivens(i,j,a,b,m):

    r = np.hypot(a,b)
    c = a / r
    s = -b / r

    G = np.eye(m)
    G[i,i] = c
    G[j,j] = c
    G[i,j] = s
    G[j,i] = -s

    return G

def QRrankKApproximation1(A,k=0,tol=1.e-3):


    if(k==0):
        [Q,R,P,k,duration] = permutedQR1(A)
    else:
        [Q,R,P,k,duration] = permutedQR1(A,k=k)

    R11 = R[0:k,0:k]
    R12 = R[0:k, k:]
    Q11 = Q[:, 0:k]

    Ak = np.block([Q11 @ R11, Q11 @ R12]) @ np.transpose(P)

    diff = np.linalg.norm(A - Ak)

    # print('Time to create QR factorization 1:', duration)
    # print('Rank ', k, 'approximation')
    # print('Error of ', diff)


    return [k, P, Q, R, Ak, Q11, R11, R12, duration]

def QRrankKApproximation2(A,k=0,tol=1.e-3):

    [m,n] = A.shape

    [Q,R,P,duration] = QRGivens(A,k=k,tol=tol)

    if(k==0):
        diagonals = np.diag(R)
        for i in range(min([m,n])):
            if(diagonals[i] < tol):
                break
            k += 1

    R11 = R[0:k,0:k]
    R12 = R[0:k, k:]
    Q11 = Q[:, 0:k]

    Ak = np.block([np.matmul(Q11,R11), np.matmul(Q11,R12)])

    diff = np.linalg.norm(A - np.matmul(Ak,np.transpose(P)))

    print('Time to create QR factorization 2:', duration)
    print('Rank ', k, 'approximation')
    print('Error of ', diff)


    return [Ak, R11, R12, Q11]

## comparison subroutines
##

def compareSVDandQR(A, precision, k=0, fullOutput=False):
    '''
    Print the time and error between the SVD and permuted QR approximations A_k 
    of some matrix A\n
    Inputs:\n
    \t  A: (m x n) matrix to approximate\n
    \t  precision: number of decimal places to display for full factorizations\n
    \t  k: (optional, default 0) desired rank of approximation, if 0 is used, 
    \t\t  a value for k will be determined automatically\n
    \t  fullOutput: (optional, default False) if True, the full factorizations 
    \t\t  will be printed\n
    Outputs:\n
    \t\t  (svdTime, qrTime, svdErr, qrErr) where:\n
    \t\t\t  svdTime: time taken for SVD factorization to run\n
    \t\t\t  qrTime: time taken for permuted QR factorization to run\n
    \t\t\t  svdErr: error between difference of A and its SVD approximation
    \t\t\t\t  using the 2-norm\n
    \t\t\t  qrErr: error between difference of A and its permuted QR 
    \t\t\t\t  factorization using the 2-norm\n
    '''

    # - calculate rank, r, of A
    r = np.linalg.matrix_rank(A)

    # make sure k <= r, otherwise we can't approximate
    if k > r:
        print("ERROR in compareSVDandQR: k not less than or equal to", \
              "rank of A")
        return(0, 0, 0, 0)
    
    # - create the SVD approximation of A and get time required to create the  
    #   SVD of A
    svdK, svdP, svdSigma, svdQT, svdAk, svdPk, svdSigmak, svdQTk, svdTime =\
        SVDrankKApproximation(A, k=k)
    
    # - create the QR approximation of A and get the time required to create the
    #   QR factorization of A
    qrK, qrP, qrQ, qrR, qrAk, qrQk, qrR11, qrR12, qrTime =\
        QRrankKApproximation1(A, k=k)
    qrPT = np.transpose(qrP)
    qrRk = np.block([qrR11, qrR12])

    # compute errors between A and its approximations
    svdErr = np.linalg.norm(A - svdAk, 2)
    qrErr = np.linalg.norm(A - qrAk, 2)

    # TODO: finish output part

    # output
    print("\nTime to compute SVD factorization of A:", str(svdTime))
    print("||A - A_K||_2 =", str(svdErr))
    print("\nTime to compute permuted QR factorization of A:", str(qrTime))
    print("||A - A_k||_2 =", str(qrErr))

    # optional output
    if fullOutput:
        # disclaimer
        print("\nDisplayed precision:", str(precision), "decimal places")
        
        # original SVD
        print("\nSVD factorization of A with rank r =", str(r))
        prettyPrintFactorization(A, svdP, svdSigma, svdQT, precision,\
                                 Aname="A", Bname="P", Cname="Sigma",\
                                    Dname="Q^T")
        
        # SVD approximation
        print("\nApproximation A_k built from SVD factorization with rank k =",\
              str(svdK))
        prettyPrintFactorization(svdAk, svdPk, svdSigmak, svdQTk,\
                                 precision, Aname="A_k", Bname="P_k",\
                                    Cname="Sigma_k", Dname="Q_k^T")
        
        # original QR
        print("\nQR factorization of A with rank r =", str(r))
        prettyPrintFactorization(A, qrQ, qrR, qrPT, precision,\
                                 Aname="A", Bname="Q", Cname="R",\
                                    Dname="P^T")
        
        # QR approximation
        print("\nApproximation A_k built from QR factorization with rank k =",\
              str(qrK))
        prettyPrintFactorization(qrAk, qrQk, qrRk, qrP, precision,\
                                 Aname="A_k", Bname="Q_k", Cname="R_k",\
                                    Dname="P^T")
        
        # print("Q_k @ R_k @ P^T")
        # print(qrQk @ qrRk @ qrPT)

    return (svdTime, qrTime, svdErr, qrErr)