from keras.models import *
from keras.layers import *
from imblearn.over_sampling import RandomOverSampler
from tkinter import *

import numpy as np
seed = 10
np.random.seed(seed)
data = np.genfromtxt('data/cs-training.csv',delimiter=",",skip_header=1)
predict_data = np.genfromtxt('data/cs-test.csv', delimiter=",", skip_header=1)
data = np.nan_to_num(data)
predict_data = np.nan_to_num(predict_data)
predict_data = predict_data[:,2:]
test_data = data[120000:,:]
data=data[:120000,:]
X=data[:,2:]
Y=data[:,1]
ros = RandomOverSampler()
X,Y=ros.fit_sample(X,Y)
X_test=test_data[:,2:]
Y_test=test_data[:,1]
X_test,Y_test = ros.fit_sample(X_test,Y_test)
features = len(X[0])
m = Sequential()
m.add(Dense(15,input_dim=features,init='uniform',activation='relu'))
m.add(Dense(15,init='uniform',activation='tanh'))
m.add(Dense(1,init='uniform',activation='sigmoid'))
m.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
m.fit(X,Y,nb_epoch=5,batch_size=100)
score = m.evaluate(X_test,Y_test)

#predictions = m.predict(predict_data)
# round predictions
#rounded = [round(x[0]) for x in predictions]
#print(predictions)
#print(rounded)

def show_entry_fields():
   f1 = float(e1.get())
   f2 = float(e2.get())
   f3 = float(e3.get())
   f4 = float(e4.get())
   f5 = float(e5.get())
   f6 = float(e6.get())
   f7 = float(e7.get())
   f8 = float(e8.get())
   f9 = float(e9.get())
   f10 = float(e10.get())
   prediction = m.predict(np.array([[f1, f2, f3, f4, f5, f6, f7, f8, f9, f10]]))
   print(prediction)
   rounded_prediction = round(prediction[0][0])
   if(rounded_prediction == 1):
       v.set("Delinquency Expected within 2 Years")
   else :
       v.set("Delinquency Not Predicted")



master = Tk()
Label(master, text="Revolving Utilization Of Unsecured Lines").grid(row=0)
Label(master, text="Age").grid(row=1)
Label(master, text="Number Of Time 30-59 Days Past Due Not Worse").grid(row=2)
Label(master, text="Debt Ratio").grid(row=3)
Label(master, text="Monthly Income").grid(row=4)
Label(master, text="Number Of Open Credit Lines And Loans").grid(row=5)
Label(master, text="Number Of Times 90 Days Late").grid(row=6)
Label(master, text="Number Real Estate Loans Or Lines").grid(row=7)
Label(master, text="Number Of Time 60-89 Days Past Due Not Worse").grid(row=8)
Label(master, text="Number Of Dependents").grid(row=9)


e1 = Entry(master)
e2 = Entry(master)
e3 = Entry(master)
e4 = Entry(master)
e5 = Entry(master)
e6 = Entry(master)
e7 = Entry(master)
e8 = Entry(master)
e9 = Entry(master)
e10 = Entry(master)

e1.grid(row=0, column=1)
e2.grid(row=1, column=1)
e3.grid(row=2, column=1)
e4.grid(row=3, column=1)
e5.grid(row=4, column=1)
e6.grid(row=5, column=1)
e7.grid(row=6, column=1)
e8.grid(row=7, column=1)
e9.grid(row=8, column=1)
e10.grid(row=9, column=1)


Button(master, text='Quit', command=master.quit).grid(row=10, column=0, sticky=W, pady=4)
Button(master, text="Serious Dlq in 2yrs?", command=show_entry_fields).grid(row=10, column=1, sticky=W, pady=4)
v = StringVar()
answer = Label(master, textvariable=v).grid(row=11)
mainloop( )