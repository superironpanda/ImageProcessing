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
        inputString = self.inputString.get("1.0", tk.END)
        oldArrayFlattened = np.fromstring(inputString, sep=',', dtype=int)
        self.RLEBitPlaneEncode = RLEBitPlane.encode(oldArrayFlattened)

        print("Run Length Encoding in Bit Plane: ")
        print(self.RLEBitPlaneEncode)
        print("Run Length Encoding Size in Bit Plane: ")
        print(sys.getsizeof(self.RLEBitPlaneEncode))
        print("Original Array size: ")
        print(sys.getsizeof(list(inputString)))


    def RLEBitPlaneDecodeButtonAction(self):
        decoded = RLEBitPlane.decode(self.RLEBitPlaneEncode)
        print("RLE in Bit Plane Decoded:")
        print(decoded)

    def RLEButtonAction(self):
        inputString = self.inputString.get("1.0", tk.END)
        #oldimagearray = self.convert_to_array()
        #oldArrayFlattened = oldimagearray.flatten()
        oldArrayFlattened = np.fromstring(inputString, sep=',', dtype=int)
        self.RLEoutput = RLE.encode(oldArrayFlattened)

        print("Run Length Encoding 1D array: ")
        print(self.RLEoutput)
        print("Run Length Encoding 1D array Size: ")
        print(sys.getsizeof(self.RLEoutput))
        print("Original Image 1D array size: ")
        print(sys.getsizeof(oldArrayFlattened))

    def RLEDecodeButtonAction(self):
        decodedArray = RLE.decode(self.RLEoutput)
        #decodedArray2D = np.reshape(decodedArray, (-1, 2))
        #self.show_new_image(decodedArray2D, "RLE Decode")
        print("RLE Decoded array: ")
        print(decodedArray)

    def huffmanEncode(self):
        inputString = self.inputString.get("1.0", tk.END)
        self.huffmanClass = HuffMan.Huffman(inputString)
        self.encodedString = self.huffmanClass.encode(inputString)

        bitsDecode = self.huffmanClass.decode(self.encodedString)
        b = bitstring.BitArray("0b" + self.encodedString)
        print("Huffman encoded string:")
        print(self.encodedString)
        print("Huffman encoded string size:")
        print(sys.getsizeof(b))
        print("Original string size:")
        print(sys.getsizeof(inputString))

    def huffmanDecode(self):
        bitsDecode = self.huffmanClass.decode(self.encodedString)
        print("Huffman decoded string:")
        print(bitsDecode)

    def LZWEncode(self):
        inputString = self.inputString.get("1.0", tk.END)
        self.LZWEncoded = LZW.compress(inputString)
        print("LZW Encoded string:")
        print(self.LZWEncoded)

    def LZWDecode(self):
        LZWDecoded = LZW.decompress(self.LZWEncoded)
        print("LZW Decoded string:")
        print(LZWDecoded)

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


root = tk.Tk()
root.minsize(600, 400)

app = Application(master=root)
app.mainloop()