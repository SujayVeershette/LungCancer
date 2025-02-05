#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
df=pd.read_csv('lungcancer.csv')
df["GENDER"] = df["GENDER"].map({"M": 0, "F": 1}) 
df["LUNG_CANCER"] = df["LUNG_CANCER"].map({"NO": 0, "YES": 1})


# ## naivebayes

# In[7]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
df.head()
x=df.drop('LUNG_CANCER',axis=1)
y=df['LUNG_CANCER']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model=GaussianNB()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
y_pred
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc*100:.2f}%")


# ## decision tree

# In[9]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
df.head()
x=df.drop('LUNG_CANCER',axis=1)
y=df['LUNG_CANCER']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model=DecisionTreeClassifier()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
y_pred
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc*100:.2f}%")


# ## Random forest

# In[10]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
df.head()
x=df.drop('LUNG_CANCER',axis=1)
y=df['LUNG_CANCER']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model=RandomForestClassifier()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
y_pred
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc*100:.2f}%")


# ## SVM

# In[11]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
df.head()
x=df.drop('LUNG_CANCER',axis=1)
y=df['LUNG_CANCER']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model=SVC()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
y_pred
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc*100:.2f}%")


# ## KNN

# In[12]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
df.head()
x=df.drop('LUNG_CANCER',axis=1)
y=df['LUNG_CANCER']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model=RandomForestClassifier()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
y_pred
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc*100:.2f}%")


# In[14]:


import matplotlib.pyplot as plt
import numpy as np

models = ["Random Forest", "KNN", "Decision Tree", "Naïve Bayes", "SVM"]
accuracies = [0.9677, 0.9677, 0.9677, 0.9516, 0.9677]

colors = ['green', 'green', 'green', 'red', 'green',]

plt.figure(figsize=(10, 5))
bars = plt.bar(models, accuracies, color=colors)


for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.02, f"{acc:.4f}", 
             ha='center', color='white', fontweight='bold')

plt.xlabel("Machine Learning Models")
plt.ylabel("Accuracy Score")
plt.title("Accuracy Comparison of Different ML Models (Test Size = 0.2)")
plt.ylim(0.9, 1.0) 
plt.grid(axis='y', linestyle="--", alpha=0.7)

plt.show()

