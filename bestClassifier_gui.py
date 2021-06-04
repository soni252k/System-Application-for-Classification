from tkinter import *
import tkinter as tk
import tkinter.font as font
from tkinter import messagebox
import time
import subprocess

import bestClassifier

root= tk.Tk()
root.title("Best Classifier")

canvas1 = tk.Canvas(root, bg="black",width = 740, height = 500)
canvas1.pack()
entry1 = tk.Entry (root, width= 45, borderwidth = '5')
canvas1.create_window(450, 50,window=entry1)

entry2 = tk.Entry (root,width= 45, borderwidth = '5')
canvas1.create_window(450, 100, window=entry2)

Label(text="   Input Dataset Link         ", fg="black", bg="yellow", font=("Helvetica", 12)).place(x=95,y=40)
Label(text=" Input Target Variable      ", fg="black", bg="yellow", font=("Helvetica", 12)).place(x=95,y=90)

# ---------------------------------------------------------------------------------------------------------------

def storeValues():
    dataset = entry1.get()
    target = entry2.get()
        
    bestClassifier.bestClassifierFunc(dataset, target)

    Label(text="Accuracy of Classifiers are given as follows :", bg="black", fg="yellow", font=("Helvetica", 12)).place(
        x=95, y=210)

    # ---------------------------------------------------------------------------------------------------------------

    Label(text=bestClassifier.Classifiers[0], bg="white", fg="black",width= 18, font=("Helvetica", 12)).place(x=95, y=250)
    Label(text=bestClassifier.Accuracy[0], bg="black", fg="white", font=("Helvetica", 12)).place(x=275, y=250)

    Label(text=bestClassifier.Classifiers[1], bg="white", fg="black", width=18, font=("Helvetica", 12)).place(x=400, y=250)
    Label(text=bestClassifier.Accuracy[1], bg="black", fg="white", font=("Helvetica", 12)).place(x=580, y=250)

    Label(text=bestClassifier.Classifiers[2], bg="white", fg="black",width=18, font=("Helvetica", 12)).place(x=95, y=300)
    Label(text=bestClassifier.Accuracy[2], bg="black", fg="white", font=("Helvetica", 12)).place(x=275, y=300)

    Label(text=bestClassifier.Classifiers[3], bg="white", fg="black", width=18,font=("Helvetica", 12)).place(x=400, y=300)
    Label(text=bestClassifier.Accuracy[3], bg="black", fg="white", font=("Helvetica", 12)).place(x=580, y=300)

    Label(text=bestClassifier.Classifiers[4], bg="white", fg="black", width=18,font=("Helvetica", 12)).place(x=95, y=350)
    Label(text=bestClassifier.Accuracy[4], bg="black", fg="white", font=("Helvetica", 12)).place(x=275, y=350)

    Label(text=bestClassifier.Classifiers[5], bg="white", fg="black", width=18, font=("Helvetica", 12)).place(x=400, y=350)
    Label(text=bestClassifier.Accuracy[5], bg="black", fg="white", font=("Helvetica", 12)).place(x=580, y=350)

    Label(root, text="The Best Classifier for our dataset is : \n" + bestClassifier.Classifiers[bestClassifier.val], fg="cyan", bg="black",
          font=("Helvetica", 20)).place(x=115, y=410)
    
    # ---------------------------------------------------------------------------------------------------------------

button1 = tk.Button(text='Submit',font=("Helvetica", 14), fg="red",bg= "cyan",  command=storeValues, width= 30, borderwidth = '10')
canvas1.create_window(350, 165, window=button1)


root.mainloop()
