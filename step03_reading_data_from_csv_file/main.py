from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from keras.utils.np_utils import to_categorical
from tkinter import *

seed = 7
np.random.seed(seed)

#http://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
#http://www.kdnuggets.com/2015/05/machine-learning-wars-amazon-google-bigml-predicsis.html
#https://mljar.com/blog/machine-learning-wars/

csv_all_data = np.genfromtxt('data/cs-training.csv', delimiter=",")
csv_predict = np.genfromtxt('data/cs-test.csv', delimiter=",")
#print(csv[1])
csv_all_data = csv_all_data[1:-1,:] #remove first name row
csv_all_data = csv_all_data[:,1:] #remove first index column
deliquent_count = np.count_nonzero(csv_all_data[:,0] > 0) #count which are deliquent
print(deliquent_count)
deliquent_rows = csv_all_data[csv_all_data[:,0] > 0]
not_deliquent_rows = csv_all_data[csv_all_data[:,0] <= 0]
print("deliquent_rows: ")
print(len(deliquent_rows))
print(len(not_deliquent_rows))
selected_not_deliquent_rows = not_deliquent_rows[0:deliquent_count,:]
print("selected_not_deliquent_rows: ")
print(len(selected_not_deliquent_rows))
all_training_data = np.concatenate((deliquent_rows, selected_not_deliquent_rows), axis=0)
np.random.shuffle(all_training_data)
print("all_training_data: ")
print(len(all_training_data))


testsize = 1000
#all_training_data = all_training_data[0:len(csv_all_data)-testsize,:]
#all_test_data = all_training_data[len(csv_all_data)-testsize:len(csv_all_data),:]

train_labels = all_training_data[:,0] #slice the first column which are the labels
train_data = all_training_data[:,np.arange(1,11)]
print(len(train_labels))
#train_labels = to_categorical(train_labels)

#test_labels = all_test_data[:,0] #slice the first column which are the labels
#test_data = all_test_data[:,np.arange(1,11)]
#test_labels = to_categorical(test_labels)

all_training_data[np.isnan(all_training_data)] = 0 #convert nan value to zero


#print(all_training_data[6])

csv_predict = csv_predict[1:-1,:] #remove first name row
csv_predict = csv_predict[:,1:] #remove first index column
csv_predict = csv_predict[:,1:] #remove empty first column
#print(csv_predict[0])



network = Sequential()
network.add(Dense(12, activation='relu', input_dim=10, kernel_initializer='uniform'))
network.add(Dense(8, kernel_initializer='uniform', activation='relu'))
network.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))


network.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

network.fit(train_data, train_labels, epochs=5, batch_size=128)

test_loss, test_acc = network.evaluate(train_data, train_labels)
print("test_acc: ")
print(test_acc)

#SeriousDlqin2yrs,RevolvingUtilizationOfUnsecuredLines,age,NumberOfTime30-59DaysPastDueNotWorse,DebtRatio,MonthlyIncome,NumberOfOpenCreditLinesAndLoans,NumberOfTimes90DaysLate,NumberRealEstateLoansOrLines,NumberOfTime60-89DaysPastDueNotWorse,NumberOfDependents

#predictions = network.predict(np.array([[0.76,45,2,0.802982129,9120,13,0,6,0,2],
#                                        [0.2, 30, 0, 0.102982129, 29120, 1, 0, 2, 0, 2],
#                                        [0.880370887, 43, 0, 0.461323764, 7523, 6, 0, 1, 0, 1],
#                                        [0.224710924,55,0,0.057234801,8700,7,0,0,0,0],
#                                        [1.135552064,41,2,0.845887215,7500,12,0,4,1,0]
#                                        ]
 #
#                                       ))

predictions = network.predict(csv_predict)
# round predictions
rounded = [round(x[0]) for x in predictions]
print(predictions)
print(rounded)



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
   prediction = network.predict(np.array([[f1, f2, f3, f4, f5, f6, f7, f8, f9, f10]]))
   print(prediction)

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
mainloop( )
