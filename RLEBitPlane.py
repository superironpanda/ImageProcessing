import numpy as np

def encode(oldArrayFlattened):
    result = []
    for i in range(8):
        encodedArray = []
        s = 0
        f = 0
        encodedArray.append(oldArrayFlattened[0] >> i & 1)
        while f < len(oldArrayFlattened):
            count = 0
            while f < len(oldArrayFlattened) and oldArrayFlattened[f] >> i & 1 == oldArrayFlattened[s] >> i & 1:
                count += 1
                f += 1
            encodedArray.append(count)
            s = f
        result.append(encodedArray)

    output = np.array(result)
    return output