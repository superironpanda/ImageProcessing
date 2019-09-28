import tkinter as tk
from tkinter import filedialog, ttk
from PIL import ImageTk, Image
import numpy as np


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
        self.BitPlaneBitsCheckBox0 = tk.Checkbutton(dropdownlisFrame, text="Bit Plane 0")
        self.BitPlaneBitsCheckBox1 = tk.Checkbutton(dropdownlisFrame, text="Bit Plane 1")
        self.BitPlaneBitsCheckBox2 = tk.Checkbutton(dropdownlisFrame, text="Bit Plane 2")
        self.BitPlaneBitsCheckBox3 = tk.Checkbutton(dropdownlisFrame, text="Bit Plane 3")
        self.BitPlaneBitsCheckBox4 = tk.Checkbutton(dropdownlisFrame, text="Bit Plane 4")
        self.BitPlaneBitsCheckBox5 = tk.Checkbutton(dropdownlisFrame, text="Bit Plane 5")
        self.BitPlaneBitsCheckBox6 = tk.Checkbutton(dropdownlisFrame, text="Bit Plane 6")
        self.BitPlaneBitsCheckBox7 = tk.Checkbutton(dropdownlisFrame, text="Bit Plane 7")

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
            newimgarray = np.asarray(self.HistogramEqualizationGlobal)
        elif chosenAlgorithmStr == "Histogram Equalization(Local)":
            method = "Histogram Equalization(Local)"
            newimgarray = np.asarray(self.HistogramEqualizationLocal)
        elif chosenAlgorithmStr == "Smoothing Filter":
            method = "Smoothing Filter"
            newimgarray = np.asarray(self.SmoothingFilter)
        elif chosenAlgorithmStr == "Median Filter":
            method = "Median Filter"
            newimgarray = np.asarray(self.MedianFilter)
        elif chosenAlgorithmStr == "Sharpening Laplacian Filter":
            method = "Sharpening Laplacian Filter"
            newimgarray = np.asarray(self.SharpeningLaplacianFilter)
        elif chosenAlgorithmStr == "High-boosting Filter":
            method = "High-boosting Filter"
            newimgarray = np.asarray(self.HighBoostingFilter)
        elif chosenAlgorithmStr == "Bit Plane":
            method = "Bit Plane"
            newimgarray = np.asarray(self.BitPlane)

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
                a = oldimgarray[y][x];
                b = oldimgarray[y][x + 1];
                c = oldimgarray[y + 1][x];
                d = oldimgarray[y + 1][x + 1];
                newimgarray[i][j] = int(a*(1-diffX)*(1-diffY)+b*diffX*(1-diffY)+c*(1-diffX)*diffY+d*diffX*diffY)
        #newimg = Image.fromarray(np.asarray(newimgarray))
        #newimg.show()
        return newimgarray

    def HistogramEqualizationGlobal(self):
        return 1

    def HistogramEqualizationLocal(self):
        return 1

    def SmoothingFilter(self):
        return 1

    def MedianFilter(self):
        return 1

    def SharpeningLaplacianFilter(self):
        return 1

    def HighBoostingFilter(self):
        return 1

    def BitPlane(self):
        return 1

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
        img = Image.open(self.filename)
        newimgarray = np.array(img)
        return newimgarray


root = tk.Tk()
root.minsize(600, 400)

app = Application(master=root)
app.mainloop()
