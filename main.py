import tkinter as tk
from tkinter import filedialog, ttk
from PIL import ImageTk, Image, ImageDraw
import numpy as np
import math
import cv2 as cv
import matplotlib.pyplot as plt

class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()
        self.create_uploadbutton()

        # Create a Tkinter variable
        tkvar = tk.StringVar(self)

        # Dictionary with options
        choices = {"None", "NN", "Bi-linear", "Bi-cubic", "Linear-X", "Linear-Y",
                   "Histogram Equalization(Local)", "Histogram Equalization(Global)",
                   "Smoothing Filter", "Median Filter", "Geometric Mean Filter",
                   "Harmonic Mean Filter", "ContraHarmonic Filter", "Max Filter",
                   "Min Filter", "Midpoint Filter", "Alpha Trimmed Filter",
                   "Sharpening Laplacian Filter", "High-boosting Filter", "Bit Plane"}
        tkvar.set("None")  # set the default option


        dropdownlisFrame = tk.LabelFrame(self)
        dropdownlisFrame.grid(column=0, row=2, padx=10, pady=10)
        popupMenu = tk.OptionMenu(dropdownlisFrame, tkvar, *choices)
        popupMenu.pack(side=tk.TOP)
        labeltest = tk.Label(dropdownlisFrame, text="Default is None")
        self.chosenAlgorithm = tkvar
        labeltest.pack(side=tk.BOTTOM)

        # on change dropdown value
        def change_dropdown(*args):
            labeltest.configure(text="The selected item is {}".format(tkvar.get()))
            self.chosenAlgorithm = format(tkvar.get())
            createFilterSizeChecker = 0
            createBitPlaneBitsChecker = 0
            createGEGlobalChecker = 0
            try:
                if self.labelFilterWidth.winfo_exists():
                    createFilterSizeChecker = 1
            except:
                createFilterSizeChecker = 0

            try:
                if self.labelBitPlaneBits.winfo_exists():
                    createBitPlaneBitsChecker = 1
            except:
                createBitPlaneBitsChecker = 0

            try:
                if self.HEGlobalLabel.winfo_exists():
                    createGEGlobalChecker = 1
            except:
                createGEGlobalChecker = 0

            if self.checkIfFilterNeededForAlgorithm() and createFilterSizeChecker == 0:
                self.createFilterTxtBox(dropdownlisFrame)
            else:
                self.destroyFilterAndBit(createFilterSizeChecker, createBitPlaneBitsChecker, createGEGlobalChecker)

            if self.checkIfBitPlaneSelected() and createBitPlaneBitsChecker == 0:
                self.createBitPlaneCheckBoxes(dropdownlisFrame)
            else:
                self.destroyFilterAndBit(createFilterSizeChecker, createBitPlaneBitsChecker, createGEGlobalChecker)

            if self.checkIfGEGlobalSelected() and createGEGlobalChecker == 0:
                self.createGEGlobalBoxes(dropdownlisFrame)
            else:
                self.destroyFilterAndBit(createFilterSizeChecker, createBitPlaneBitsChecker, createGEGlobalChecker)

        # link function to change dropdown
        tkvar.trace('w', change_dropdown)
        self.create_text_box()
        self.create_submit_button()

    def checkIfGEGlobalSelected(self):
        checkString = self.chosenAlgorithm
        if checkString == "Histogram Equalization(Global)":
            return True
        else:
            return False

    def createGEGlobalBoxes(self, dropdownlisFrame):
        self.HEGlobalLabel = tk.Label(dropdownlisFrame, text="HE Global output selection")
        self.HEGlobalLabel.pack()
        self.HEGlobalVar0 = tk.BooleanVar()
        self.HEGlobalVar1 = tk.BooleanVar()
        self.HEGlobalVar2 = tk.BooleanVar()
        self.HEGlobalCheckBox0 = tk.Checkbutton(dropdownlisFrame, text="Regular HE", var=self.HEGlobalVar0)
        self.HEGlobalCheckBox1 = tk.Checkbutton(dropdownlisFrame, text="Bit Plane 4,5,6,7", var=self.HEGlobalVar1)
        self.HEGlobalCheckBox2 = tk.Checkbutton(dropdownlisFrame, text="Bit Plane 0,1,2,3", var=self.HEGlobalVar2)
        self.HEGlobalCheckBox0.pack()
        self.HEGlobalCheckBox1.pack()
        self.HEGlobalCheckBox2.pack()

    def destroyGEGlobalBoxes(self):
        print("123")
        self.HEGlobalLabel.destroy()
        self.HEGlobalCheckBox0.destroy()
        self.HEGlobalCheckBox1.destroy()
        self.HEGlobalCheckBox2.destroy()

    def createBitPlaneCheckBoxes(self, dropdownlisFrame):
        self.labelBitPlaneBits = tk.Label(dropdownlisFrame, text="Bit Plane Bits")
        self.labelBitPlaneBits.pack()
        self.BitCheckBox0Var = tk.BooleanVar()
        self.BitCheckBox1Var = tk.BooleanVar()
        self.BitCheckBox2Var = tk.BooleanVar()
        self.BitCheckBox3Var = tk.BooleanVar()
        self.BitCheckBox4Var = tk.BooleanVar()
        self.BitCheckBox5Var = tk.BooleanVar()
        self.BitCheckBox6Var = tk.BooleanVar()
        self.BitCheckBox7Var = tk.BooleanVar()
        self.BitPlaneBitsCheckBox0 = tk.Checkbutton(dropdownlisFrame, text="Bit Plane 0", var=self.BitCheckBox0Var)
        self.BitPlaneBitsCheckBox1 = tk.Checkbutton(dropdownlisFrame, text="Bit Plane 1", var=self.BitCheckBox1Var)
        self.BitPlaneBitsCheckBox2 = tk.Checkbutton(dropdownlisFrame, text="Bit Plane 2", var=self.BitCheckBox2Var)
        self.BitPlaneBitsCheckBox3 = tk.Checkbutton(dropdownlisFrame, text="Bit Plane 3", var=self.BitCheckBox3Var)
        self.BitPlaneBitsCheckBox4 = tk.Checkbutton(dropdownlisFrame, text="Bit Plane 4", var=self.BitCheckBox4Var)
        self.BitPlaneBitsCheckBox5 = tk.Checkbutton(dropdownlisFrame, text="Bit Plane 5", var=self.BitCheckBox5Var)
        self.BitPlaneBitsCheckBox6 = tk.Checkbutton(dropdownlisFrame, text="Bit Plane 6", var=self.BitCheckBox6Var)
        self.BitPlaneBitsCheckBox7 = tk.Checkbutton(dropdownlisFrame, text="Bit Plane 7", var=self.BitCheckBox7Var)

        self.BitCheckBoxShowHistogramVar = tk.BooleanVar()
        self.BitCheckBoxShowHistogram = tk.Checkbutton(dropdownlisFrame, text="Show histogram", var=self.BitCheckBoxShowHistogramVar)
        self.BitCheckBoxShowHistogram.pack()

        self.BitPlaneBitsCheckBox0.pack()
        self.BitPlaneBitsCheckBox1.pack()
        self.BitPlaneBitsCheckBox2.pack()
        self.BitPlaneBitsCheckBox3.pack()
        self.BitPlaneBitsCheckBox4.pack()
        self.BitPlaneBitsCheckBox5.pack()
        self.BitPlaneBitsCheckBox6.pack()
        self.BitPlaneBitsCheckBox7.pack()

    def destroyFilterAndBit(self, createFilterSizeChecker, createBitPlaneBitsChecker, createGEGlobalChecker):
        if createFilterSizeChecker == 1:
            checker = self.checkIfFilterNeededForAlgorithm()
            if checker is not True:
                self.destroyFilterSizeLabelAndTextBox()

        if createBitPlaneBitsChecker == 1:
            checker = self.checkIfBitPlaneSelected()
            if checker is not True:
                self.destroyBitPlaneBitsCheckBoxes()

        if createGEGlobalChecker == 1:
            checker = self.checkIfGEGlobalSelected()
            if checker is not True:
                self.destroyGEGlobalBoxes()

    def createFilterTxtBox(self, dropdownlisFrame):
        self.labelFilterWidth = tk.Label(dropdownlisFrame, text="Filter Width")
        self.labelFilterWidth.pack()
        self.filterWidth = tk.Text(dropdownlisFrame, height=1, width=15)
        self.filterWidth.insert("1.0", "3")
        self.filterWidth.pack()

        self.labelFilterHeight = tk.Label(dropdownlisFrame, text="Filter Height")
        self.labelFilterHeight.pack()
        self.filterHeight = tk.Text(dropdownlisFrame, height=1, width=15)
        self.filterHeight.insert("1.0", "3")
        self.filterHeight.pack()

        if(self.chosenAlgorithm == "High-boosting Filter"):
            self.A = tk.Label(dropdownlisFrame, text="k")
            self.A.pack()
            self.AText = tk.Text(dropdownlisFrame, height=1, width=15)
            self.AText.pack()

        if(self.chosenAlgorithm == "ContraHarmonic Filter"):
            self.Q = tk.Label(dropdownlisFrame, text="Q")
            self.Q.pack()
            self.QText = tk.Text(dropdownlisFrame, height=1, width=15)
            self.QText.pack()

        if(self.chosenAlgorithm == "Alpha Trimmed Filter"):
            self.d = tk.Label(dropdownlisFrame, text="d")
            self.d.pack()
            self.dText = tk.Text(dropdownlisFrame, height=1, width=15)
            self.dText.pack()

    def destroyBitPlaneBitsCheckBoxes(self):
        self.labelBitPlaneBits.destroy()
        self.BitPlaneBitsCheckBox0.destroy()
        self.BitPlaneBitsCheckBox1.destroy()
        self.BitPlaneBitsCheckBox2.destroy()
        self.BitPlaneBitsCheckBox3.destroy()
        self.BitPlaneBitsCheckBox4.destroy()
        self.BitPlaneBitsCheckBox5.destroy()
        self.BitPlaneBitsCheckBox6.destroy()
        self.BitPlaneBitsCheckBox7.destroy()
        self.BitCheckBoxShowHistogram.destroy()

    def checkIfFilterNeededForAlgorithm(self):
        checkerString = self.chosenAlgorithm
        if (checkerString == "Histogram Equalization(Local)" or checkerString == "Smoothing Filter" or
            checkerString == "Median Filter" or checkerString == "Sharpening Laplacian Filter" or
            checkerString == "High-boosting Filter" or checkerString == "Geometric Mean Filter" or
            checkerString == "Harmonic Mean Filter" or checkerString == "ContraHarmonic Filter" or
            checkerString == "Max Filter" or checkerString == "Min Filter" or checkerString == "Midpoint Filter"
            or checkerString == "Alpha Trimmed Filter"):
            return True
        else:
            return False

    def checkIfBitPlaneSelected(self):
        checkString = self.chosenAlgorithm
        if checkString == "Bit Plane":
            return True
        else:
            return False

    def destroyFilterSizeLabelAndTextBox(self):
        self.labelFilterWidth.destroy()
        self.filterWidth.destroy()
        self.labelFilterHeight.destroy()
        self.filterHeight.destroy()
        try:
            self.A.destroy()
            self.AText.destroy()
            self.Q.destroy()
            self.QText.destroy()
            self.d.destroy()
            self.dText.destroy()
        except:
            a=1

    def create_text_box(self):
        labelWidth = tk.Label(self, text="Width \n(Empty would be original size)")
        labelWidth.grid(column=0, row=9)
        self.width = tk.Text(self, height=1, width=15)
        self.width.grid(column=0, row=10)

        labelHeight = tk.Label(self, text="Height \n(Empty would be original size)")
        labelHeight.grid(column=0, row=14)
        self.height = tk.Text(self, height=1, width=15)
        self.height.grid(column=0, row=15)

        labelBit = tk.Label(self, text="Bit")
        labelBit.grid(column=0, row=19)
        self.bit = tk.Text(self, height=1, width=15)
        self.bit.grid(column=0, row=20)

    def create_submit_button(self):
        self.submitButton = tk.Button(self, text="Generate", command=self.submitButtonAction)
        self.submitButton.grid(column=0, row=25, padx=10, pady=10)

    def submitButtonAction(self):
        chosenAlgorithmStr = self.chosenAlgorithm
        newimgarray = self.convert_to_array()
        method = ""
        if chosenAlgorithmStr == "None":
            print("No algorithm chosen")
        elif chosenAlgorithmStr == "NN":
            method = "Nearest Neighbor"
            newimgarray = np.asarray(self.nearest_neighbor())
        elif chosenAlgorithmStr == "Bi-linear":
            method = "Bi-linear"
            newimgarray = np.asarray(self.bilinear())
        elif chosenAlgorithmStr == "Linear-X":
            method = "Linear-X"
            newimgarray = np.asarray(self.linear_x())
        elif chosenAlgorithmStr == "Linear-Y":
            method = "Linear-Y"
            newimgarray = np.asarray(self.linear_y())
        elif chosenAlgorithmStr == "Histogram Equalization(Global)":
            method = "Histogram Equalization(Global)"
            oldimgarray = self.convert_to_array()
            newimgarray = np.asarray(self.HistogramEqualizationGlobal(oldimgarray))
        elif chosenAlgorithmStr == "Histogram Equalization(Local)":
            method = "Histogram Equalization(Local)"
            newimgarray = np.asarray(self.HistogramEqualizationLocal())
        elif chosenAlgorithmStr == "Smoothing Filter":
            method = "Smoothing Filter"
            newimgarray = np.asarray(self.SmoothingFilter())
        elif chosenAlgorithmStr == "Median Filter":
            method = "Median Filter"
            newimgarray = np.asarray(self.MedianFilter())
        elif chosenAlgorithmStr == "Sharpening Laplacian Filter":
            method = "Sharpening Laplacian Filter"
            newimgarray = np.asarray(self.SharpeningLaplacianFilter())
        elif chosenAlgorithmStr == "High-boosting Filter":
            method = "High-boosting Filter"
            newimgarray = np.asarray(self.HighBoostingFilter())
        elif chosenAlgorithmStr == "Bit Plane":
            method = "Bit Plane"
            newimgarray = np.asarray(self.BitPlane())
        elif chosenAlgorithmStr == "Geometric Mean Filter":
            method = "Geometric Mean Filter"
            newimgarray = np.asarray(self.GeometricMeanFilter())
        elif chosenAlgorithmStr == "Harmonic Mean Filter":
            method = "Harmonic Mean Filter"
            newimgarray = np.asarray(self.HarmonicMeanFilter())
        elif chosenAlgorithmStr == "ContraHarmonic Filter":
            method = "ContraHarmonic Filter"
            newimgarray = np.asarray(self.ContraHarmonicFilter())
        elif chosenAlgorithmStr == "Max Filter":
            method = "Max Filter"
            newimgarray = np.asarray(self.MaxFilter())
        elif chosenAlgorithmStr == "Min Filter":
            method = "Min Filter"
            newimgarray = np.asarray(self.MinFilter())
        elif chosenAlgorithmStr == "Midpoint Filter":
            method = "Midpoint Filter"
            newimgarray = np.asarray(self.MidpointFilter())
        elif chosenAlgorithmStr == "Alpha Trimmed Filter":
            method = "Alpha Trimmed Filter"
            newimgarray = np.asarray(self.AlphaTrimmedFilter())

        if self.bit.get("1.0", tk.END) == "\n":
            print("bits are not changing")
        else:
            newheight = int(newimgarray.shape[0])
            newwidth = int(newimgarray.shape[1])
            newBit = self.bit.get("1.0", tk.END)
            new = pow(2, int(newBit))
            ConversionFactor = 255 / (new - 1)
            for i in range(newheight):
                for j in range(newwidth):
                    newimgarray[i, j] = int(newimgarray[i, j]/ConversionFactor + 0.5)*ConversionFactor
            method = method + self.bit.get("1.0", tk.END) + " Image"
        #newimg = Image.fromarray(np.asarray(newimgarray))
        #newimg.show()
        if(method == "None"):
            method = self.bit.get("1.0", tk.END)+" Image"
        self.show_new_image(newimgarray, method)

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

    def nearest_neighbor(self):
        oldimgarray = self.convert_to_array()

        originalheight = int(oldimgarray.shape[0])
        originalwidth = int(oldimgarray.shape[1])
        newheight = int(self.height.get("1.0", tk.END))
        newwidth = int(self.width.get("1.0", tk.END))

        newimgarray = np.ndarray(shape=(newheight,
                                        newwidth), dtype=np.int)
        xr = originalwidth*1.0/newwidth
        yr = originalheight*1.0/newheight
        for i in range(newheight):
            for j in range(newwidth):
                x = int(j*xr)
                y = int(i*yr)
                newimgarray[i][j] = oldimgarray[y][x]
        """
        """
        #newimg = Image.fromarray(np.asarray(newimgarray))
        #newimg.show()
        return newimgarray

    def linear_x(self):
        oldimgarray = self.convert_to_array()

        originalheight = int(oldimgarray.shape[0])
        originalwidth = int(oldimgarray.shape[1])
        newheight = int(self.height.get("1.0", tk.END))
        newwidth = int(self.width.get("1.0", tk.END))

        newimgarray = np.ndarray(shape=(newheight,
                                        newwidth), dtype=np.int)
        xr = (originalwidth - 1) * 1.0 / newwidth
        yr = (originalheight) * 1.0 / newheight
        for i in range(newheight):
            for j in range(newwidth):
                x = int(j * xr)
                y = int(i * yr)
                diffX = (j * xr) - x
                a = oldimgarray[y][x];
                b = oldimgarray[y][x + 1];
                #newimgarray[i][j] = ((b-a)*i)+(a-i*(b-a))
                newimgarray[i][j] = int(a * (1 - diffX) + b * diffX)
        # newimg = Image.fromarray(np.asarray(newimgarray))
        # newimg.show()
        return newimgarray

    def linear_y(self):
        oldimgarray = self.convert_to_array()

        originalheight = int(oldimgarray.shape[0])
        originalwidth = int(oldimgarray.shape[1])
        newheight = int(self.height.get("1.0", tk.END))
        newwidth = int(self.width.get("1.0", tk.END))

        newimgarray = np.ndarray(shape=(newheight,
                                        newwidth), dtype=np.int)
        xr = (originalwidth) * 1.0 / newwidth
        yr = (originalheight-1) * 1.0 / newheight
        for i in range(newheight):
            for j in range(newwidth):
                x = int(j * xr)
                y = int(i * yr)
                diffY = (i * yr) - y
                a = oldimgarray[y][x]
                b = oldimgarray[y+1][x]
                # newimgarray[i][j] = ((b-a)*i)+(a-i*(b-a))
                newimgarray[i][j] = int(a * (1 - diffY) + b * diffY)
        # newimg = Image.fromarray(np.asarray(newimgarray))
        # newimg.show()
        return newimgarray

    def bilinear(self):
        oldimgarray = self.convert_to_array()

        originalheight = int(oldimgarray.shape[0])
        originalwidth = int(oldimgarray.shape[1])
        newheight = int(self.height.get("1.0", tk.END))
        newwidth = int(self.width.get("1.0", tk.END))

        newimgarray = np.ndarray(shape=(newheight,
                                        newwidth), dtype=np.int)
        xr = (originalwidth-1)*1.0/newwidth
        yr = (originalheight-1)*1.0/newheight
        for i in range(newheight):
            for j in range(newwidth):
                x = int(j*xr)
                y = int(i*yr)
                diffX = (j * xr) - x
                diffY = (i * yr) - y
                a = oldimgarray[y][x]
                b = oldimgarray[y][x + 1]
                c = oldimgarray[y + 1][x]
                d = oldimgarray[y + 1][x + 1]
                newimgarray[i][j] = int(a*(1-diffX)*(1-diffY)+b*diffX*(1-diffY)+c*(1-diffX)*diffY+d*diffX*diffY)
        #newimg = Image.fromarray(np.asarray(newimgarray))
        #newimg.show()
        return newimgarray

    def HistogramEqualizationGlobal(self, oldimgarray):
        oldimgarray1D = oldimgarray.flatten()
        hist = self.get_histogram(oldimgarray1D, 256)
        cs = self.cumsum(hist)

        # numerator & denomenator
        nj = (cs - cs.min()) * 255
        N = cs.max() - cs.min()

        # re-normalize the cdf
        cs = nj / N
        cs = cs.astype('uint8')
        img_new = cs[oldimgarray1D]
        img_new = np.reshape(img_new, oldimgarray.shape)

        height = img_new.shape[0]
        width = img_new.shape[1]
        if self.HEGlobalVar0.get():
            return img_new
        elif self.HEGlobalVar1.get():
            newimgarray = np.zeros((height, width))
            bitPlaneArray = np.array([0, 0, 0, 0, 1, 1, 1, 1])
            bitPlaneString = ""
            for i in range(8):
                bitPlaneString = str(bitPlaneArray[i]) + bitPlaneString
            # bitPlaneString = "0b"+bitPlaneString
            for i in range(height):
                for j in range(width):
                    value = img_new[i][j]
                    if int(value) & int(bitPlaneString):
                        newimgarray[i][j] = img_new[i][j] & int(bitPlaneString)
            return newimgarray
        elif self.HEGlobalVar2.get():
            newimgarray = np.zeros((height, width))
            bitPlaneArray = np.array([1, 1, 1, 1, 0, 0, 0, 0])
            bitPlaneString = ""
            for i in range(8):
                bitPlaneString = str(bitPlaneArray[i]) + bitPlaneString
            # bitPlaneString = "0b"+bitPlaneString
            for i in range(height):
                for j in range(width):
                    value = img_new[i][j]
                    if int(value) & int(bitPlaneString):
                        newimgarray[i][j] = img_new[i][j] & int(bitPlaneString)
            return newimgarray

    def get_histogram(self, image, bins):
        # array with size of bins, set to zeros
        histogram = np.zeros(bins)

        # loop through pixels and sum up counts of pixels
        for pixel in image:
            histogram[pixel] += 1

        # return our final result
        return histogram

    def cumsum(self, a):
        a = iter(a)
        b = [next(a)]
        for i in a:
            b.append(b[-1] + i)
        return np.array(b)

    def HistogramEqualizationLocal(self):
        oldimgarray = self.convert_to_array()
        sizex = int(oldimgarray.shape[0])
        sizey = int(oldimgarray.shape[1])
        filtersize = int(self.filterHeight.get("1.0", tk.END))
        paddedimg = self.addpadding(oldimgarray, filtersize)
        newimg = np.asarray(np.zeros((sizex, sizey)))
        filter_size = np.array([filtersize, filtersize])
        edge = int(math.floor(filtersize/2))

        for i in range(0, sizex):
            for j in range(0, sizey):
                kernel = self.HELocalFillKernel(i, j, filter_size, oldimgarray)
                EqualizedKernel = self.HistogramEqualizationGlobal(kernel)
                newimg[i][j] = self.HELocalFindCenterPixel(filtersize, EqualizedKernel)
        return newimg

    def HELocalFindCenterPixel(self, kernelsize, equalizedKernel):
        tmpEqualizedKernel = equalizedKernel.flatten()
        return tmpEqualizedKernel[int(kernelsize*kernelsize/2)]

    def HELocalFillKernel(self, i, j, kernelsize, image):
        kernel = np.zeros((kernelsize[0], kernelsize[1]), dtype=int)
        center = int(kernelsize[0]/2)
        h = image.shape[0]
        w = image.shape[1]
        pixel = int(0)
        for y in range(kernelsize[0]):
            for x in range(kernelsize[0]):
                deltaY = y - center
                deltaX = x - center
                if i+deltaY>=0 and i+deltaY<h and j+deltaX>=0 and j+deltaX<w:
                    pixel = int(image[i+deltaY][j+deltaX])
                else:
                    pixel = 0
                kernel[y][x]=pixel
        return kernel

    def HELocalFillImagewithEqualizedKernel(self, newimg, EqualizedKernel, i, j, kernelsize):
        from_x = i - int(math.floor(kernelsize / 2))
        to_x = i + int(math.floor(kernelsize / 2))
        from_y = j - int(math.floor(kernelsize / 2))
        to_y = j + int(math.floor(kernelsize / 2))
        newimg[from_x:to_x+1, from_y:to_y+1] = EqualizedKernel

        return newimg

    def addpadding(self, source, pad):
        imarr = np.array(source)
        padimarr = np.zeros((imarr.shape[0] + 2 * pad, imarr.shape[1] + 2 * pad), dtype=np.uint8)
        padimarr[pad:padimarr.shape[0] - pad, pad:padimarr.shape[1] - pad] = imarr
        return padimarr

    def SmoothingFilter(self):
        image = self.convert_to_array()
        filtertmp=int(self.filterHeight.get("1.0", tk.END))
        filter_size = np.array([filtertmp, filtertmp])
        filtered_image = image
        num_rows = image.shape[1]
        num_cols = image.shape[0]
        newimg = np.asarray(np.zeros((num_cols, num_rows)))
        # assumes the kernel is simmetric and of odd dimensions
        edge = int(math.floor(filter_size[0]/2))

        for j in range(0, num_rows):
            for i in range(0, num_cols):
                kernel = self.fill_kernel(i, j, filter_size, image)
                newimg[i, j] = self.get_mean(kernel.flatten())

        return newimg

    def MaxFilter(self):
        image = self.convert_to_array()
        filtertmp = int(self.filterHeight.get("1.0", tk.END))
        filter_size = np.array([filtertmp, filtertmp])
        filtered_image = image
        num_rows = image.shape[1]
        num_cols = image.shape[0]
        newimg = np.asarray(np.zeros((num_cols, num_rows)))
        # assumes the kernel is simmetric and of odd dimensions
        edge = int(math.floor(filter_size[0] / 2))

        for j in range(0, num_rows):
            for i in range(0, num_cols):
                kernel = self.fill_kernel(i, j, filter_size, image)
                newimg[i, j] = self.getMax(kernel.flatten())

        return newimg

    def getMax(self, vector):
        max = 0
        for i in range(len(vector)):
            if(int(vector[i]) > max):
                max = vector[i]
        return max

    def MinFilter(self):
        image = self.convert_to_array()
        filtertmp = int(self.filterHeight.get("1.0", tk.END))
        filter_size = np.array([filtertmp, filtertmp])
        filtered_image = image
        num_rows = image.shape[1]
        num_cols = image.shape[0]
        newimg = np.asarray(np.zeros((num_cols, num_rows)))
        # assumes the kernel is simmetric and of odd dimensions
        edge = int(math.floor(filter_size[0] / 2))

        for j in range(0, num_rows):
            for i in range(0, num_cols):
                kernel = self.fill_kernel(i, j, filter_size, image)
                newimg[i, j] = self.getMin(kernel.flatten())

        return newimg

    def getMin(self, vector):
        min = 0
        for i in range(len(vector)):
            if vector[i] < min:
                min = vector[i]

        return min

    def MidpointFilter(self):
        image = self.convert_to_array()
        filtertmp = int(self.filterHeight.get("1.0", tk.END))
        filter_size = np.array([filtertmp, filtertmp])
        filtered_image = image
        num_rows = image.shape[1]
        num_cols = image.shape[0]
        newimg = np.asarray(np.zeros((num_cols, num_rows)))
        # assumes the kernel is simmetric and of odd dimensions
        edge = int(math.floor(filter_size[0] / 2))

        for j in range(0, num_rows):
            for i in range(0, num_cols):
                kernel = self.fill_kernel(i, j, filter_size, image)
                newimg[i, j] = self.getMidPoint(kernel.flatten())

        return newimg

    def getMidPoint(self, vector):
        midPoint = 0
        max = self.getMax(vector)
        min = self.getMin(vector)
        midPoint = int((max+min)/2)

        return midPoint

    def AlphaTrimmedFilter(self):
        image = self.convert_to_array()
        filtertmp = int(self.filterHeight.get("1.0", tk.END))
        filter_size = np.array([filtertmp, filtertmp])
        filtered_image = image
        num_rows = image.shape[1]
        num_cols = image.shape[0]
        newimg = np.asarray(np.zeros((num_cols, num_rows)))
        # assumes the kernel is simmetric and of odd dimensions
        edge = int(math.floor(filter_size[0] / 2))

        for j in range(0, num_rows):
            for i in range(0, num_cols):
                kernel = self.fill_kernel(i, j, filter_size, image)
                newimg[i, j] = self.getAlphaTrimmed(kernel.flatten())

        return newimg

    def getAlphaTrimmed(self, vector):
        sum = 0
        d = int(self.dText.get("1.0", tk.END))
        for i in range(int(d/2), int(len(vector)-d/2)):
            sum += vector[i]
        result = (1/(len(vector))) * sum

        return int(result)

    def HarmonicMeanFilter(self):
        image = self.convert_to_array()
        filtertmp=int(self.filterHeight.get("1.0", tk.END))
        filter_size = np.array([filtertmp, filtertmp])
        filtered_image = image
        num_rows = image.shape[1]
        num_cols = image.shape[0]
        newimg = np.asarray(np.zeros((num_cols, num_rows)))
        # assumes the kernel is simmetric and of odd dimensions
        edge = int(math.floor(filter_size[0]/2))

        for j in range(0, num_rows):
            for i in range(0, num_cols):
                kernel = self.fill_kernel(i, j, filter_size, image)
                newimg[i, j] = self.getHarmonicMean(kernel.flatten())

        return newimg

    def ContraHarmonicFilter(self):
        image = self.convert_to_array()
        filtertmp = int(self.filterHeight.get("1.0", tk.END))
        filter_size = np.array([filtertmp, filtertmp])
        filtered_image = image
        num_rows = image.shape[1]
        num_cols = image.shape[0]
        newimg = np.asarray(np.zeros((num_cols, num_rows)))
        # assumes the kernel is simmetric and of odd dimensions
        edge = int(math.floor(filter_size[0] / 2))

        for j in range(0, num_rows):
            for i in range(0, num_cols):
                kernel = self.fill_kernel(i, j, filter_size, image)
                newimg[i, j] = self.getContraHarmonicMean(kernel.flatten())

        return newimg

    def getContraHarmonicMean(self, vector):
        Q = self.QText.get("1.0", tk.END)
        valueQ = float(Q)
        mean = np.float64(0)
        top = np.float64(0)
        bottom = np.float64(0)

        for i in range(len(vector)):
            mean += (pow(vector[i], valueQ+1)/pow(vector[i], valueQ))

        if math.isnan(mean):
            mean = 0

        if int(mean)>255:
            mean = 255
        elif int(mean) <0:
            mean=0

        return int(mean)

    def GeometricMeanFilter(self):
        image = self.convert_to_array()
        filtertmp=int(self.filterHeight.get("1.0", tk.END))
        filter_size = np.array([filtertmp, filtertmp])
        filtered_image = image
        num_rows = image.shape[1]
        num_cols = image.shape[0]
        newimg = np.asarray(np.zeros((num_cols, num_rows)))
        # assumes the kernel is simmetric and of odd dimensions
        edge = int(math.floor(filter_size[0]/2))

        for j in range(0, num_rows):
            for i in range(0, num_cols):
                kernel = self.fill_kernel(i, j, filter_size, image)
                newimg[i, j] = self.getProduct(kernel.flatten())
        return newimg

    def getProduct(self, vector):
        product = np.float64(1)
        lengthVector = int(len(vector))
        for i in range(0, lengthVector):
            product *= vector[i]

        product = int(pow(product, (1/lengthVector)))
        if int(product) > 255:
            product = 255
        elif int(product) < 0:
            product = 0

        return int(product)

    def getHarmonicMean(self, vector):
        mean = np.float64(0)
        for i in range(0, len(vector)):
            mean += (1/vector[i])
        return int(len(vector) / mean)

    def fill_kernel_Product(self, i, j, kernel_size, image):
        kernel = np.zeros((kernel_size[0], kernel_size[1]), dtype=int)
        center = int(kernel_size[0] / 2)
        h = image.shape[0]
        w = image.shape[1]
        pixel = int(0)
        for y in range(kernel_size[0]):
            for x in range(kernel_size[0]):
                deltaY = y - center
                deltaX = x - center
                if i + deltaY >= 0 and i + deltaY < h and j + deltaX >= 0 and j + deltaX < w:
                    pixel = int(image[i + deltaY][j + deltaX])
                else:
                    pixel = int(image[i][j])
                kernel[y][x] = pixel
        return kernel

    def get_mean(self, vector):
        mean = 0
        for i in range(0, len(vector)):
            mean += vector[i]
        return int(mean / len(vector))

    def fill_kernel(self, i, j, kernel_size, image):
        kernel = np.zeros((kernel_size[0], kernel_size[1]), dtype=int)
        center = int(kernel_size[0] / 2)
        h = image.shape[0]
        w = image.shape[1]
        pixel = int(0)
        for y in range(kernel_size[0]):
            for x in range(kernel_size[0]):
                deltaY = y - center
                deltaX = x - center
                if i + deltaY >= 0 and i + deltaY < h and j + deltaX >= 0 and j + deltaX < w:
                    pixel = int(image[i + deltaY][j + deltaX])
                else:
                    pixel = int(image[i][j])
                kernel[y][x] = pixel
        return kernel

    def MedianFilter(self):
        image = self.convert_to_array()
        filtertmp = int(self.filterHeight.get("1.0", tk.END))
        filter_size = np.array([filtertmp, filtertmp])

        num_rows = image.shape[1]
        num_cols = image.shape[0]
        filtered_image = np.asarray(np.zeros((num_cols, num_rows)))
        # asumes the kernel is simmetric and of odd dimensions
        edge = int(math.floor(filter_size[0] / 2))

        for j in range(0, num_rows):
            for i in range(0, num_cols):
                kernel = self.fill_kernel(i, j, filter_size, image)
                filtered_image[i, j] = self.get_median(kernel.flatten())

        return filtered_image

    def get_median(self, vector):
        vector = np.sort(vector)
        median = vector[int(math.floor(len(vector) / 2))]
        return median

    def SharpeningLaplacianFilter(self):
        img = self.convert_to_array()
        filtertmp = int(self.filterHeight.get("1.0", tk.END))
        filter_size = np.array([filtertmp, filtertmp])

        num_rows = img.shape[1]
        num_cols = img.shape[0]
        newimg = np.asarray(np.zeros((num_cols, num_rows)))
        #imgwithpadding = self.addpadding(image, filtertmp)
        mask = self.createLaplacianKernel(filtertmp)
        counter = 0
        for j in range(0, num_rows):
            for i in range(0, num_cols):
                kernel = self.fill_kernel(i, j, filter_size, img)
                newimg[i, j] = self.findLaplacianCenterValue(mask, kernel, filtertmp, 0)

        for j in range(0, num_rows):
            for i in range(0, num_cols):
                newValue = newimg[i, j] + img[i, j]
                if newValue > 255:
                    newValue = 255
                newimg[i, j] = newValue
        return newimg

    def testfunc(self):
        print(self.kernel)

    def findLaplacianCenterValue(self, mask, kernel, filtersize, A):
        p = int(0)
        for i in range(filtersize):
            for j in range(filtersize):
                kernelValue = kernel[i][j]
                maskValue = int(mask[i][j])
                if i == int(filtersize/2) and j == int(filtersize/2):
                    p += int(maskValue*(kernelValue+A))
                else:
                    p += int(maskValue*kernelValue)
        if p > 255:
            p = 255
        elif p < 0:
            p = 0
        return p

    def createLaplacianKernel(self, filtersize):
        mask = np.ones((filtersize, filtersize))
        center = -(filtersize*filtersize - 1)
        mask[int(filtersize/2)][int(filtersize/2)] = center
        return np.asarray(mask)

    def createHightBoostingFilter(self, filtersize):
        mask = np.ones((filtersize, filtersize))
        center = (filtersize * filtersize - 1)
        for i in range(filtersize):
            for j in range(filtersize):
                mask[i, j] = -1
        mask[int(filtersize / 2)][int(filtersize / 2)] = center
        return np.asarray(mask)
        return mask

    def HighBoostingFilter(self):
        blurredImg = self.SmoothingFilter()
        originalImg = self.convert_to_array()
        newheight = int(originalImg.shape[0])
        newwidth = int(originalImg.shape[1])
        gmask = np.ndarray(shape=(newheight,
                                  newwidth), dtype=np.int)
        newimgarray = np.ndarray(shape=(newheight,
                                        newwidth), dtype=np.int)
        A = int(self.AText.get("1.0", tk.END))
        for i in range(newheight):
            for j in range(newwidth):
                value = int(originalImg[i][j]) - int(blurredImg[i][j])
                if value > 255:
                    value = 255
                elif value < 0:
                    value = 0
                gmask[i][j] = value
        for i in range(newheight):
            for j in range(newwidth):
                value = int(originalImg[i][j] + A * gmask[i][j])
                if value > 255:
                    value = 255
                elif value < 0:
                    value = 0
                newimgarray[i][j] = value
        return newimgarray

    def BitPlane(self):
        oldimagearray = self.convert_to_array()
        height = int(oldimagearray.shape[0])
        width = int(oldimagearray.shape[1])
        bitPlane1 = np.zeros((height, width))
        bitPlane2 = np.zeros((height, width))
        bitPlane3 = np.zeros((height, width))
        bitPlane4 = np.zeros((height, width))
        bitPlane5 = np.zeros((height, width))
        bitPlane6 = np.zeros((height, width))
        bitPlane7 = np.zeros((height, width))
        bitPlane8 = np.zeros((height, width))
        newimgarray = np.zeros((height, width))
        bitPlaneArray = np.array(np.zeros(8), dtype=np.int)

        if self.BitCheckBox0Var.get():
            bitPlaneArray[0] = 1
        if self.BitCheckBox1Var.get():
            bitPlaneArray[1] = 1
        if self.BitCheckBox2Var.get():
            bitPlaneArray[2] = 1
        if self.BitCheckBox3Var.get():
            bitPlaneArray[3] = 1
        if self.BitCheckBox4Var.get():
            bitPlaneArray[4] = 1
        if self.BitCheckBox5Var.get():
            bitPlaneArray[5] = 1
        if self.BitCheckBox6Var.get():
            bitPlaneArray[6] = 1
        if self.BitCheckBox7Var.get():
            bitPlaneArray[7] = 1
        bitPlaneString = ""
        for i in range(8):
            bitPlaneString = str(bitPlaneArray[i]) + bitPlaneString
        #bitPlaneString = "0b"+bitPlaneString
        for i in range(height):
            for j in range(width):
                value = oldimagearray[i][j]
                if int(value) & int(bitPlaneString):
                    newimgarray[i][j] = oldimagearray[i][j] & int(bitPlaneString, 2)

        if self.BitCheckBoxShowHistogramVar.get():
            histogram = newimgarray.flatten()
            b, bins, patches = plt.hist(histogram, 255)
            plt.xlim([0, 255])
            plt.show()
            '''plt.hist(newimgarray.ravel(), 256, [0, 256])
            plt.show()'''

        return newimgarray

    def AddBitPlaneToFinalImage(self, height, width, newimgarray, bitplaneimg):
        for i in range(height):
            for j in range(width):
                    newimgarray[i][j] += bitplaneimg[i][j]
        return newimgarray

    def create_uploadbutton(self):
        self.labelFrame = ttk.LabelFrame(self, text="Open A File")
        self.labelFrame.grid(column=0, row=0, padx=10, pady=10)
        self.uploadButton = ttk.Button(self.labelFrame, text="Browse A File", command=self.uploadButtonAction)
        self.uploadButton.grid(column=0, row=1)

    def uploadButtonAction(self):
        self.filename = filedialog.askopenfilename()
        self.label = ttk.Label(self.labelFrame, text="")
        self.label.grid(column=1, row=2)
        self.label.configure(text=self.filename)
        self.show_original_image()

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

    def convert_to_array(self):
        img = Image.open(self.filename).convert('L')
        newimgarray = np.array(img)
        return newimgarray


root = tk.Tk()
root.minsize(600, 400)

app = Application(master=root)
app.mainloop()