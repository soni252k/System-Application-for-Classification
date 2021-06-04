from tkinter import *
import tkinter.font as font
from tkinter import messagebox
import time
import subprocess
root=Tk()

root.geometry("7400x500")
root.title("Classification Algorithms")
root.minsize(740,500)
root.maxsize(740, 500)
root.configure(bg="black")

#---------------------------------------------------------------------------------------------------------------

#options

def informationGain():
    subprocess.call(['python','informationGain_gui.py'])


def giniIndex():
    subprocess.call(['python','giniIndex_gui.py'])


def naiveBayes():
    subprocess.call(['python','naiveBayes_gui.py'])


def knn():
    subprocess.call(['python','knn_gui.py'])


def randomForest():
    subprocess.call(['python','randomForest_gui.py'])


def gradientBoost():
    subprocess.call(['python','gradientBoost_gui.py'])


def bestClassifier():
    subprocess.call(['python','bestClassifier_gui.py'])
    
#---------------------------------------------------------------------------------------------------------------
def credit():
    messagebox.showinfo(title="Credits", message='''                    B. Tech.   CSE 3rd Year

                    Kaushiki Taru      -   18103053
                    Kumari Soni        -   18103054
                    

Dr. B. R. Ambedkar National Instiitute of Technology''')





class App(Frame):
    def __init__(self,master=None):
        Frame.__init__(self, master)
        self.master = master
        self.label = Label(text="", fg="cyan2",bg="black", font=("Helvetica", 18))
        self.label.place(x=315,y=2)
        self.update_clock()

    def update_clock(self):
        now = time.strftime("%H:%M:%S")
        self.label.configure(text=now)
        self.after(1000, self.update_clock)

app = App(root)
root.after(1000, app.update_clock)

heading1 = Label(text="Main Screen" ,bg="black", fg="white", font=("Helvetica", 20))
heading1.place(x=285,y=35)
heading = Label(text="Classification Algorithms with GUI" ,bg="black", fg="white", font=("Helvetica", 20))
heading.place(x=200,y=70)


heading = Label(text="Press respective button to implement desired algorithm " ,bg="black", fg="cyan2", font=("Helvetica", 13))
heading.place(x=170,y=120)

myFont = font.Font(weight="bold",size=11)

#---------------------------------------------------------------------------------------------------------------

#buttons functioning 
b1 = Button(root,width=30,bg="azure2", fg="black", text="Information Gain",borderwidth = '4',command=informationGain, relief=RAISED )
b1['font'] = myFont
b1.place(x=95,y=160)



b2 = Button(root,width=30,bg="azure2", fg="black", text="Gini Index",borderwidth = '4',command=giniIndex, relief=RAISED )
b2['font'] = myFont
b2.place(x=400,y=160)



b3 = Button(root, bg="azure2", fg="black",width=30, text="Naive Bayes",borderwidth = '4', command=naiveBayes, relief=RAISED)
b3['font'] = myFont
b3.place(x=95,y=210)



b4 = Button(root, bg="azure2", fg="black",width=30,  text="K- Nearest Neighbour",borderwidth = '4', command=knn, relief=RAISED)
b4['font'] = myFont
b4.place(x=400,y=210)



b5 = Button(root, bg="azure2", fg="black",width=30, text="Random Forest",borderwidth = '4', command=randomForest, relief=RAISED)
b5['font'] = myFont
b5.place(x=95,y=260)



b6 = Button(root, bg="azure2", fg="black",width=30, text="Gradient Boost",borderwidth = '4', command=gradientBoost, relief=RAISED)
b6['font'] = myFont
b6.place(x=400,y=260)



b7 = Button(root,width=64,bg="azure2", fg="black", text="Know the Best Classifier",borderwidth = '4',command=bestClassifier, relief=RAISED )
b7['font'] = myFont
b7.place(x=95,y=310)

#---------------------------------------------------------------------------------------------------------------

b8 = Button(root, bg="azure2", fg="black",width=64, text="Credits",borderwidth = '20', command=credit)
b8['font'] = myFont
b8.place(x=85,y=360)

#---------------------------------------------------------------------------------------------------------------


#Label(text="Kumari Soni, 18103054" ,bg="black", fg="white", font=("Helvetica", 12)).place(x=360,y=380)
#Label(text="Kaushiki Taru, 18103053" ,bg="black", fg="white", font=("Helvetica", 12)).place(x=360,y=355)

#---------------------------------------------------------------------------------------------------------------

root.mainloop()




