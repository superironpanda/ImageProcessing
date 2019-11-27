from HuffMan import Huffman
import sys
import bitstring

testString = "ababddcabadsf"
test = Huffman(testString)
bits = test.encode(testString)

bitsDecode = test.decode(bits)
b = bitstring.BitArray("0b"+bits)

print(sys.getsizeof(testString))


print(b)
print(sys.getsizeof(b))
