import numpy as np
import RLE

def encode(oldArrayFlattened):
    # result will lead with 0 bit plane
    result = []
    binaryOldArray = []
    # convert original numbers to 8 bit binary
    for i in range(len(oldArrayFlattened)):
        binaryOldArray.append(format(oldArrayFlattened[i], '08b'))

    bitPlane = []

    for i in range(8):
        for j in range(len(binaryOldArray)):
            bitPlane.append(binaryOldArray[j][i])
        result.append(bitPlane)
        bitPlane=[]

    result = np.asarray(result, dtype=int)
    resultEncoded = []
    for i in range(8):
        tmp = result[i]
        tmpEncoded = RLE.encode(tmp)
        resultEncoded.append(list(tmpEncoded))

    return resultEncoded

def decode(encodedList):
    result = []
    for i in range(8):
        tmp = list(RLE.decode(np.asarray(encodedList[i], dtype=int)))
        result.append(tmp)
    resultDecoded = []
    tmp = []
    for i in range(len(result[0])):
        for j in range(8):
            tmp.append(result[j][i])
        resultDecoded.append(tmp)
        tmp = []
    resultFinished = []
    for i in range(len(resultDecoded)):
        tmp = resultDecoded[i]
        num = ''.join(map(str, tmp))
        resultFinished.append(int(num, 2))

    return resultFinished

