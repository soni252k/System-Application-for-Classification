from tkinter import *
import tkinter as tk
import tkinter.font as font
from tkinter import messagebox
import time
import subprocess

import naiveBayes

root= tk.Tk()
root.title("Naive Bayes")

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
    naiveBayes.naiveBayesFunc(dataset, target)

    Label(text="Result of Naive Bayes Classifier :", bg="black", fg="yellow", font=("Helvetica", 12)).place(
        x=95, y=210)

    # ---------------------------------------------------------------------------------------------------------------

    Label(text="Accuracy", bg="white", fg="black",width= 18, font=("Helvetica", 12)).place(x=95, y=250)
    Label(text=naiveBayes.Accuracy, bg="black", fg="white", font=("Helvetica", 12)).place(x=275, y=250)

    Label(text="Precision", bg="white", fg="black", width=18, font=("Helvetica", 12)).place(x=400, y=250)
    Label(text=naiveBayes.Precision, bg="black", fg="white", font=("Helvetica", 12)).place(x=580, y=250)

    Label(text="Recall", bg="white", fg="black",width=18, font=("Helvetica", 12)).place(x=95, y=300)
    Label(text=naiveBayes.Recall, bg="black", fg="white", font=("Helvetica", 12)).place(x=275, y=300)

    Label(text="F1 Score", bg="white", fg="black", width=18,font=("Helvetica", 12)).place(x=400, y=300)
    Label(text=naiveBayes.F1Score, bg="black", fg="white", font=("Helvetica", 12)).place(x=580, y=300)

    Label(text="Confusion Matrix", bg="white", fg="black", width=18,font=("Helvetica", 12)).place(x=245, y=350)
    Label(text=naiveBayes.CM, bg="black", fg="white", font=("Helvetica", 12)).place(x=300, y=400)
    
    # ---------------------------------------------------------------------------------------------------------------

button1 = tk.Button(text='Submit',font=("Helvetica", 14), fg="red",bg= "cyan",  command=storeValues, width= 30, borderwidth = '10')
canvas1.create_window(350, 165, window=button1)


root.mainloop()
