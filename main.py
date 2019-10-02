import tkinter as tk
from tkinter import filedialog, ttk
from PIL import ImageTk, Image, ImageDraw
import numpy as np
import math
import cv2 as cv


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
                   "Smoothing Filter", "Median Filter",
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

            if self.checkIfFilterNeededForAlgorithm() and createFilterSizeChecker == 0:
                self.createFilterTxtBox(dropdownlisFrame)
            else:
                self.destroyFilterAndBit(createFilterSizeChecker, createBitPlaneBitsChecker)

            if self.checkIfBitPlaneSelected() and createBitPlaneBitsChecker == 0:
                self.createBitPlaneCheckBoxes(dropdownlisFrame)
            else:
                self.destroyFilterAndBit(createFilterSizeChecker, createBitPlaneBitsChecker)

        # link function to change dropdown
        tkvar.trace('w', change_dropdown)
        self.create_text_box()
        self.create_submit_button()

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

        self.BitPlaneBitsCheckBox0.pack()
        self.BitPlaneBitsCheckBox1.pack()
        self.BitPlaneBitsCheckBox2.pack()
        self.BitPlaneBitsCheckBox3.pack()
        self.BitPlaneBitsCheckBox4.pack()
        self.BitPlaneBitsCheckBox5.pack()
        self.BitPlaneBitsCheckBox6.pack()
        self.BitPlaneBitsCheckBox7.pack()

    def destroyFilterAndBit(self, createFilterSizeChecker, createBitPlaneBitsChecker):
        if createFilterSizeChecker == 1:
            checker = self.checkIfFilterNeededForAlgorithm()
            if checker is not True:
                self.destroyFilterSizeLabelAndTextBox()

        if createBitPlaneBitsChecker == 1:
            checker = self.checkIfBitPlaneSelected()
            if checker is not True:
                self.destroyBitPlaneBitsCheckBoxes()

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

    def checkIfFilterNeededForAlgorithm(self):
        checkerString = self.chosenAlgorithm
        if (checkerString == "Histogram Equalization(Local)" or checkerString == "Smoothing Filter" or
            checkerString == "Median Filter" or checkerString == "Sharpening Laplacian Filter" or
            checkerString == "High-boosting Filter"):
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
                a = oldimgarray[y][x];
                b = oldimgarray[y+1][x];
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

        return img_new

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
        newimg = np.asarray(np.zeros((num_rows, num_cols)))
        # assumes the kernel is simmetric and of odd dimensions
        edge = int(math.floor(filter_size[0]/2))

        for j in range(0, num_rows):
            for i in range(0, num_cols):
                kernel = self.fill_kernel(i, j, filter_size, image)
                newimg[i, j] = self.get_mean(kernel.flatten())

        return newimg

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
                    pixel = 0
                kernel[y][x] = pixel
        return kernel

    def MedianFilter(self):
        image = self.convert_to_array()
        filtertmp = int(self.filterHeight.get("1.0", tk.END))
        filter_size = np.array([filtertmp, filtertmp])

        num_rows = image.shape[1]
        num_cols = image.shape[0]
        filtered_image = np.asarray(np.zeros((num_rows, num_cols)))
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
        newimg = np.asarray(np.zeros((num_rows, num_cols)))
        #imgwithpadding = self.addpadding(image, filtertmp)
        mask = self.createLaplacianKernel(filtertmp)
        counter = 0
        for j in range(0, num_rows):
            for i in range(0, num_cols):
                self.kernel = self.fill_kernel(i, j, filter_size, img)
                newimg[i, j] = self.findLaplacianCenterValue(mask, 2, filtertmp)
        return newimg

    def testfunc(self):
        print(self.kernel)

    def findLaplacianCenterValue(self, mask, kernel, filtersize):
        p = int(0)
        for i in range(filtersize):
            for j in range(filtersize):
                kernelValue = self.kernel[i][j]
                p += int(mask[i][j]*kernelValue)
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

    def HighBoostingFilter(self):
        blurredImg = self.SmoothingFilter()
        originalImg = self.convert_to_array()
        newheight = int(originalImg.shape[0])
        newwidth = int(originalImg.shape[1])
        gmask = np.ndarray(shape=(newheight,
                                        newwidth), dtype=np.int)
        newimgarray = np.ndarray(shape=(newheight,
                                        newwidth), dtype=np.int)
        for i in range(newheight):
            for j in range(newwidth):
                gmask[i][j] = int(originalImg[i][j]) - int(blurredImg[i][j])
        for i in range(newheight):
            for j in range(newwidth):
                newimgarray[i][j] = int(originalImg[i][j] + 1.2*gmask[i][j])
        return gmask

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
        for i in range(height):
            for j in range(width):
                value = oldimagearray[i][j]
                if value < 2:
                    bitPlane1[i][j] = oldimagearray[i][j]
                elif value < 4:
                    bitPlane2[i][j] = oldimagearray[i][j]
                elif value < 8:
                    bitPlane3[i][j] = oldimagearray[i][j]
                elif value < 16:
                    bitPlane4[i][j] = oldimagearray[i][j]
                elif value < 32:
                    bitPlane5[i][j] = oldimagearray[i][j]
                elif value < 64:
                    bitPlane6[i][j] = oldimagearray[i][j]
                elif value < 128:
                    bitPlane7[i][j] = oldimagearray[i][j]
                elif value < 256:
                    bitPlane8[i][j] = oldimagearray[i][j]

        if self.BitCheckBox0Var.get():
            newimgarray = self.AddBitPlaneToFinalImage(height, width, newimgarray, bitPlane1)
        if self.BitCheckBox1Var.get():
            newimgarray = self.AddBitPlaneToFinalImage(height, width, newimgarray, bitPlane2)
        if self.BitCheckBox2Var.get():
            newimgarray = self.AddBitPlaneToFinalImage(height, width, newimgarray, bitPlane3)
        if self.BitCheckBox3Var.get():
            newimgarray = self.AddBitPlaneToFinalImage(height, width, newimgarray, bitPlane4)
        if self.BitCheckBox4Var.get():
            newimgarray = self.AddBitPlaneToFinalImage(height, width, newimgarray, bitPlane5)
        if self.BitCheckBox5Var.get():
            newimgarray = self.AddBitPlaneToFinalImage(height, width, newimgarray, bitPlane6)
        if self.BitCheckBox6Var.get():
            newimgarray = self.AddBitPlaneToFinalImage(height, width, newimgarray, bitPlane7)
        if self.BitCheckBox7Var.get():
            newimgarray = self.AddBitPlaneToFinalImage(height, width, newimgarray, bitPlane8)

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
