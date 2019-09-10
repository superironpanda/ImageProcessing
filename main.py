import tkinter as tk
from tkinter import filedialog, ttk
from PIL import ImageTk, Image
import numpy as np
import math

class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()
        self.create_uploadbutton()

        # Create a Tkinter variable
        tkvar = tk.StringVar(self)

        # Dictionary with options
        choices = {"None", "NN", "L", "Bi-linear", "Bi-cubic", "Linear-X", "Linear-Y"}
        tkvar.set("None")  # set the default option


        dropdownlisFrame = tk.LabelFrame(self)
        dropdownlisFrame.grid(column=0, row=2, padx=10, pady=10)
        popupMenu = tk.OptionMenu(dropdownlisFrame, tkvar, *choices)
        popupMenu.pack(side= tk.TOP)
        labeltest = tk.Label(dropdownlisFrame, text="Default is None")
        self.chosenAlgorithm = tkvar
        labeltest.pack(side=tk.BOTTOM)

        # on change dropdown value
        def change_dropdown(*args):
            labeltest.configure(text="The selected item is {}".format(tkvar.get()))
            self.chosenAlgorithm = format(tkvar.get())

        # link function to change dropdown
        tkvar.trace('w', change_dropdown)
        self.create_text_box()
        self.create_submit_button()

    def create_text_box(self):
        labelWidth = tk.Label(self, text="Width")
        labelWidth.grid(column=0, row=9)
        self.width = tk.Text(self, height=1, width=15)
        self.width.grid(column=0, row=10)

        labelHeight = tk.Label(self, text="Width")
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
        if(chosenAlgorithmStr == "None"):
            print("No algorithm chosen")
        elif(chosenAlgorithmStr == "NN"):
            method = "Nearest Neighbor"
            newimgarray = np.asarray(self.nearest_neighbor())
        elif(chosenAlgorithmStr == "Bi-linear"):
            method = "Bi-linear"
            newimgarray = np.asarray(self.bilinear())
        elif (chosenAlgorithmStr == "Linear-X"):
            method = "Linear-X"
            newimgarray = np.asarray(self.linear_x())
        elif (chosenAlgorithmStr == "Linear-Y"):
            method = "Linear-Y"
            newimgarray = np.asarray(self.linear_y())

        if (self.bit.get("1.0", tk.END) == "\n"):
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
        newimgarray = Image.fromarray(np.asarray(args))
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

    def create_uploadbutton(self):
        self.labelFrame = ttk.LabelFrame(self, text="Open A File")
        self.labelFrame.grid(column=0, row=0, padx=10, pady=10)
        self.uploadButton = ttk.Button(self.labelFrame, text="Browse A File", command=self.uploadButtonAction)
        self.uploadButton.grid(column=0, row=1)

    def uploadButtonAction(self):
        self.filename = filedialog.askopenfilename(initialdir="/", title="Select A File",
                                                   filetype=(("PGM", "*.pgm"), ("All Files", "*.*")))
        self.label = ttk.Label(self.labelFrame, text="")
        self.label.grid(column=1, row=2)
        self.label.configure(text=self.filename)
        self.show_original_image()

    def show_original_image(self):
        """
        self.canvas = tk.Canvas(root, width=300, height=300)
        self.canvas.pack()
        self.img = ImageTk.PhotoImage(Image.open(self.filename))
        self.canvas.create_image(10, 10, anchor=tk.NW, image=self.img)
        self.canvas.image = self.img
        """
        #img = Image.open(self.filename)

        #img.show()
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
        #newimg = Image.fromarray(newimgarray)
        #newimg.show()
        return newimgarray


root = tk.Tk()
root.minsize(600, 400)

app = Application(master=root)
app.mainloop()
