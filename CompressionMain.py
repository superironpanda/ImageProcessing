import sys
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import ImageTk, Image, ImageDraw
import numpy as np
import RLE
import RLEBitPlane
import HuffMan
import bitstring
import LZW
import time

class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()
        self.create_uploadbutton()
        self.createRLEButtons()
        self.createTextBox()
        self.createRLEBitPlaneButtons()
        self.createHuffmanButtons()
        self.createLZWButtons()

    def createTextBox(self):
        self.inputString = tk.Text(height=1, width=15)

        self.inputString.pack()

    def create_uploadbutton(self):
        self.labelFrame = ttk.LabelFrame(self, text="Open A File")
        self.labelFrame.grid(column=0, row=3, padx=10, pady=10)
        self.uploadButton = ttk.Button(self.labelFrame, text="Browse A File", command=self.uploadButtonAction)
        self.uploadButton.grid(column=0, row=4)

    def createHuffmanButtons(self):
        self.huffmanButton = tk.Button(self, text="Huffman Encoding", command=self.huffmanEncode)
        self.huffmanButton.grid(column=0, row=50, padx=10, pady=10)
        self.huffmanDecodeButton = tk.Button(self, text="Huffman Decoding", command=self.huffmanDecode)
        self.huffmanDecodeButton.grid(column=2, row=50, padx=10, pady=10)

    def createLZWButtons(self):
        self.LZWButton = tk.Button(self, text="LZW Encoding", command=self.LZWEncode)
        self.LZWButton.grid(column=0, row=60, padx=10, pady=10)
        self.LZWDecodeButton = tk.Button(self, text="LZW Decoding", command=self.LZWDecode)
        self.LZWDecodeButton.grid(column=2, row=60, padx=10, pady=10)

    def uploadButtonAction(self):
        self.filename = filedialog.askopenfilename()
        self.label = ttk.Label(self.labelFrame, text="")
        self.label.grid(column=1, row=2)
        self.label.configure(text=self.filename)
        self.show_original_image()

    def createRLEButtons(self):
        self.RLEButton = tk.Button(self, text="Run Length Encoding", command=self.RLEButtonAction)
        self.RLEButton.grid(column=0, row=30, padx=10, pady=10)
        self.RLEDecodeButton = tk.Button(self, text="Run Length Decoding", command=self.RLEDecodeButtonAction)
        self.RLEDecodeButton.grid(column=2, row=30, padx=10, pady=10)

    def createRLEBitPlaneButtons(self):
        self.RLEBitPlaneButton = tk.Button(self, text="RLE Bit Plane Encoding", command=self.RLEBitPlaneButtonAction)
        self.RLEBitPlaneButton.grid(column=0, row=40, padx=10, pady=10)
        self.RLEBitPlaneDecodeButton = tk.Button(self, text="RLE Bit Plane Decoding", command=self.RLEBitPlaneDecodeButtonAction)
        self.RLEBitPlaneDecodeButton.grid(column=2, row=40, padx=10, pady=10)

    def show_original_image(self):
        img = ImageTk.PhotoImage(Image.open(self.filename))
        newWindow = tk.Toplevel(root)

        newWindow.title("Original Image")
        newWindow.geometry("512x512")
        newWindow.configure(background='grey')

        newWindow.canvas = tk.Canvas(newWindow, width=800, height=800)

        newWindow.img = ImageTk.PhotoImage(Image.open(self.filename))
        newWindow.canvas.create_image(10, 10, anchor=tk.NW, image=img)
        newWindow.canvas.image = img
        newWindow.canvas.pack()
        newWindow.mainloop()

    def RLEBitPlaneButtonAction(self):
        oldimagearray = self.convert_to_array()
        oldArrayFlattened = oldimagearray.flatten()
        start = time.time()
        self.RLEBitPlaneEncode = RLEBitPlane.encode(oldArrayFlattened)
        end = time.time()
        self.RLEBitPlaneEncode = np.asarray(self.RLEBitPlaneEncode)
        execTime = self.compressionRatio(sys.getsizeof(self.RLEBitPlaneEncode), sys.getsizeof(oldArrayFlattened))
        #oldArrayFlattened = oldArrayFlattened.astype(np.uint16)
        print("-----------------------------------------------------------")
        print("Run Length Encoding Size in Bit Plane: ")
        print(sys.getsizeof(self.RLEBitPlaneEncode))
        print("Encoded type (numpy array with lists):")
        print(self.RLEBitPlaneEncode.dtype)
        print("Original Array size: ")
        print(sys.getsizeof(oldArrayFlattened))
        print("old array type:")
        print(oldArrayFlattened.dtype)
        print("Compression Ratio:")
        print(str(execTime) + "%")
        print("Execution Time:")
        print(end-start)
        print("-----------------------------------------------------------")


    def RLEBitPlaneDecodeButtonAction(self):
        oldimagearray = self.convert_to_array()
        start = time.time()
        decoded = np.asarray(RLEBitPlane.decode(self.RLEBitPlaneEncode))
        end = time.time()
        decoded = decoded.reshape((len(oldimagearray), len(oldimagearray[0])))
        print("-----------------------------------------------------------")
        print("Decode Execution Time:")
        print(end-start)
        print("-----------------------------------------------------------")
        self.show_new_image(decoded, "RLE Bit Plane Decoded")

    def RLEButtonAction(self):
        #inputString = self.inputString.get("1.0", tk.END)
        oldimagearray = self.convert_to_array()
        oldArrayFlattened = oldimagearray.flatten()
        #oldArrayFlattened = np.fromstring(inputString, sep=',', dtype=int)
        startTime = time.time()
        self.RLEoutput = RLE.encode(oldArrayFlattened)
        endTime = time.time()
        oldArrayFlattened = oldArrayFlattened.astype(np.uint16)
        compressionRatio = self.compressionRatio(sys.getsizeof(self.RLEoutput), sys.getsizeof(oldArrayFlattened))
        print("-----------------------------------------------------------")
        print("Run Length Encoding 1D array Size: ")
        print(sys.getsizeof(self.RLEoutput))
        print("Encoded array type:")
        print(self.RLEoutput.dtype)
        print("Original Image 1D array size: ")
        print(sys.getsizeof(oldArrayFlattened))
        print("old array type:")
        print((oldArrayFlattened.dtype))
        print("Compression Ratio:")
        print(str(compressionRatio) + "%")
        print("Execution Time:")
        print(endTime-startTime)
        print("-----------------------------------------------------------")

    def RLEDecodeButtonAction(self):
        oldimagearray = self.convert_to_array()
        startTime = time.time()
        decodedArray = RLE.decode(self.RLEoutput)
        endTime = time.time()
        #decodedArray2D = np.reshape(decodedArray, (-1, 2))
        #self.show_new_image(decodedArray2D, "RLE Decode")

        B = decodedArray.reshape((len(oldimagearray), len(oldimagearray[0])))
        print("-----------------------------------------------------------")
        print("Decode Time:")
        print(endTime-startTime)
        print("-----------------------------------------------------------")
        self.show_new_image(B, "RLE Decoded")

    def huffmanEncode(self):
        oldimagearray = self.convert_to_array()
        oldArrayFlattened = list(oldimagearray.flatten())
        start = time.time()
        self.hm = HuffMan.Picture(self.filename)
        self.hm.load_data()
        self.hm.make_heap()
        self.hm.merge_nodes()
        self.hm.heaporder(self.hm.heap[0], "")
        self.hm.create_compression_keys()
        self.HMencoded = self.hm.writeout()
        self.hm.readin()
        #tmp = " ".join(str(e) for e in self.HMencoded)

        #b = bitstring.BitArray("0b" + tmp)
        end = time.time()
        compR = self.compressionRatio(sys.getsizeof(self.HMencoded), sys.getsizeof(oldArrayFlattened))
        print("-----------------------------------------------------------")
        print("Huffman encoded string size:")
        print(sys.getsizeof(self.HMencoded))
        print("Encoded type:")
        print(type(self.HMencoded))
        print("Original string size:")
        print(sys.getsizeof(oldArrayFlattened))
        print("Original type:")
        print(type(oldArrayFlattened))
        print("Compression Ratio:")
        print(str(compR)+"%")
        print("Execution Time:")
        print(end-start)
        print("-----------------------------------------------------------")

    def huffmanDecode(self):
        oldimagearray = self.convert_to_array()
        start = time.time()
        decodedImg = self.hm.create_new_image()
        end = time.time()
        decodedImg.show()
        print("-----------------------------------------------------------")
        print("Decode execution time:")
        print(end - start)
        print("-----------------------------------------------------------")

    def LZWEncode(self):
        start = time.time()
        inputString = self.inputString.get("1.0", tk.END)
        self.LZWEncoded, self.LZWDictionary = LZW.compress(inputString)
        end = time.time()
        inputString = list(inputString)
        compR = self.compressionRatio(sys.getsizeof(self.LZWEncoded), sys.getsizeof(inputString))
        print("-----------------------------------------------------------")
        print("LZW Encoded string:")
        print(self.LZWEncoded)
        print("Encoded size:")
        print(sys.getsizeof(self.LZWEncoded))
        print("Original size:")
        print(sys.getsizeof(inputString))
        print("Compression Ratio:")
        print(str(compR) + "%")
        print("LZW Execution time:")
        print(end - start)
        print("Dictionary:")
        print(self.LZWDictionary)
        print("-----------------------------------------------------------")

    def LZWDecode(self):
        start = time.time()
        LZWDecoded = LZW.decompress(self.LZWEncoded, self.LZWDictionary)
        end = time.time()
        print("-----------------------------------------------------------")
        print("LZW Decoded string:")
        print(LZWDecoded)
        print("LZW Execution time:")
        print(end - start)
        print("-----------------------------------------------------------")

    def convert_to_array(self):
        img = Image.open(self.filename).convert('L')
        newimgarray = np.array(img)
        return newimgarray

    def show_new_image(self, args, method):
        newimgarray = Image.fromarray(np.asarray(args, np.uint8))
        img = ImageTk.PhotoImage(Image.fromarray(np.asarray(newimgarray)))
        newWindow = tk.Toplevel(root)

        newWindow.title(method)
        newWindow.geometry("512x512")
        newWindow.configure(background='grey')

        newWindow.canvas = tk.Canvas(newWindow, width=800, height=800)

        newWindow.img = ImageTk.PhotoImage(Image.open(self.filename))
        newWindow.canvas.create_image(10, 10, anchor=tk.NW, image=img)
        newWindow.canvas.image = img
        newWindow.canvas.pack()
        newWindow.mainloop()

    def compressionRatio(self, new, old):
        return (new/old) * 100


root = tk.Tk()
root.minsize(600, 400)

app = Application(master=root)
app.mainloop()