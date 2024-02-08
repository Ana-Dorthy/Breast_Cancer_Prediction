#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# In[2]:


df=pd.read_csv("data.csv")


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df.isna().sum()


# In[6]:


df.shape


# In[7]:


df.fillna(0, inplace=True)


# In[8]:


df.shape


# In[9]:


df.describe()


# In[10]:


df['diagnosis'].value_counts()


# In[11]:


sns.countplot(x='diagnosis', data=df, label="count")

plt.show()


# In[12]:


from sklearn.preprocessing import LabelEncoder
labelencoder_Y = LabelEncoder()
df[df.columns[1]] = labelencoder_Y.fit_transform(df.iloc[:,1].values)
df.head()


# In[13]:


sns.pairplot(df.iloc[:,1:5],hue="diagnosis")


# In[14]:


X=df.iloc[:,2:32].values 
Y=df.iloc[:,1].values 


# In[15]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)


# In[16]:


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_cost(X, y, theta):
    m = len(y)
    h = sigmoid(X.dot(theta))
    cost = (1 / m) * np.sum((-y * np.log(h)) - ((1 - y) * np.log(1 - h)))
    return cost

def gradient_descent(X, y, theta, alpha, num_iterations):
    m = len(y)
    costs = []

    for _ in range(num_iterations):
        h = sigmoid(X.dot(theta))
        gradient = (1 / m) * X.T.dot(h - y)
        theta -= alpha * gradient
        cost = compute_cost(X, y, theta)
        costs.append(cost)

    return theta, costs

X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))

theta = np.zeros(X_train.shape[1])

alpha = 0.01
num_iterations = 1000

# Perform gradient descent
theta, costs = gradient_descent(X_train, Y_train, theta, alpha, num_iterations)

# Predict labels for training data
X_train_prediction = sigmoid(X_train.dot(theta))
X_train_prediction = np.round(X_train_prediction).astype(int)


# In[17]:


training_data_accuracy = np.mean(X_train_prediction == Y_train)

print('accuracy on training data =', training_data_accuracy)

X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))

X_test_prediction = sigmoid(X_test.dot(theta))
X_test_prediction = np.round(X_test_prediction).astype(int)

testing_data_accuracy = np.mean(X_test_prediction == Y_test)

print('accuracy on testing data =', testing_data_accuracy)


# In[18]:


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict(input_data, theta):
    input_data = np.hstack((np.ones((input_data.shape[0], 1)), input_data))
    predictions = sigmoid(input_data.dot(theta))
    predictions = np.round(predictions).astype(int)
    return predictions

input_data = np.array([17.14,16.4,116,912.7,0.1186,0.2276,0.2229,0.1401,0.304,0.07413,1.046,0.976,7.276,111.4,0.008029,0.03799,0.03732,0.02397,0.02308,0.007444,22.25,21.4,152.4,1461,0.1545,0.3949,0.3853,0.255,0.4066,0.1059])

input_data_reshaped = input_data.reshape(1, -1)

predictions = predict(input_data_reshaped, theta)

if predictions[0] == 1:
    print('The breast tumor is Malignant')
else:
    print('The breast tumor is Benign')


# In[ ]:


import tkinter as tk
from tkinter import messagebox,ttk
import numpy as np

root = tk.Tk()
root.title("Breast Cancer Classification")
root.geometry("1500x400")
root.configure(bg="#FFD6E7")
heading_label = tk.Label(root, text="Breast Cancer Prediction", font=("Helvetica", 20, "bold"), bg="#FFD6E7", fg="#333333")
heading_label.pack(pady=20)
input_frame = tk.Frame(root, bg="#FFD6E7")
input_frame.pack(pady=20)


input_labels = ['Radius_mean', 'Texture_mean', 'Perimeter_mean', 'Area_mean', 'Smoothness_mean', 'Compactness_mean', 
                'Concavity_mean', 'Concave_points_mean', 'Symmetry_mean','Fractal_dimension_mean', 'Radius_se', 'Texture_se',
                'Perimeter_se', 'Area_se', 'Smoothness_se', 'Compactness_se', 'Concavity_se', 'Concave_points_se',          
                'Symmetry_se', 'Fractal_dimension_se', 'Radius_worst', 'Texture_worst', 'Perimeter_worst', 'Area_worst', 
                'Smoothness_worst', 'Compactness_worst', 'Concavity_worst', 'Concave_points_worst', 'Symmetry_worst', 
                'Fractal_dimension_worst']

input_entries = []
for i, label in enumerate(input_labels):
    row = i // 5
    col = i % 5
    
    label = tk.Label(input_frame, text=label + ":", bg="#FFD6E7", fg="#333333", font=("Helvetica", 12))
    label.grid(row=row, column=2*col, sticky="e", padx=5, pady=5)

    entry = tk.Entry(input_frame, width=10, font=("Helvetica", 12))
    entry.grid(row=row, column=2*col+1, padx=5, pady=5)
    input_entries.append(entry)

def predict_cancer():
    input_data = [float(entry.get()) for entry in input_entries]
    input_data = np.array(input_data)

    input_data_reshaped = input_data.reshape(1, -1)

    predictions = predict(input_data_reshaped, theta)

    if predictions[0] == 1:
        messagebox.showinfo("Prediction Result", "The breast tumor is Malignant", icon="warning")
    else:
        messagebox.showinfo("Prediction Result", "The breast tumor is Benign", icon="info")


style = ttk.Style()
style.configure("TButton",
                foreground="#FFFFFF",
                background="#FF69B4",
                font=("Helvetica", 12),
                padding=(10, 5))

predict_button = ttk.Button(root, text="Classify", command=predict_cancer, style="TButton")
predict_button.pack(pady=10)
root.mainloop()

