import numpy as np

def encode(oldArrayFlattened):

    newArrayStr = []
    f = 0
    s = 0
    while f < len(oldArrayFlattened):
        count = 0
        while (f < len(oldArrayFlattened) and oldArrayFlattened[f] == oldArrayFlattened[s]):
            count += 1
            f += 1
        tmp = []
        newArrayStr.append(count)
        newArrayStr.append(oldArrayFlattened[s])
        s = f
    newImgArray = np.asarray(newArrayStr)
    output = newImgArray.astype(np.uint8)
    return output

def decode(encodedArray):
    output = []
    times = 0
    i = 0
    value = 0
    while i < len(encodedArray):
        value = encodedArray[i+1]
        times = encodedArray[i]
        while times > 0:
            output.append(value)
            times-=1
        i+=2
    outputArray = np.asarray(output)
    return outputArray

