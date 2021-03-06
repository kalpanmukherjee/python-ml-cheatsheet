# Python - Machine Learning CheatSheet

## Neural Networks
```python
from keras.models import Sequential
from keras.layers import Dense, Dropout

classifier.add(Dense(20, activation='relu', kernel_initializer='random_normal', input_dim=x_train.shape[1]))
classifier.add(Dense(30, activation='relu', kernel_initializer='random_normal'))
classifier.add(Dropout(0.2))
classifier.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal'))

classifier.compile(optimizer ='adam',loss='binary_crossentropy', metrics =['accuracy'])

classifier.fit(x_train, y_train, batch_size=100, epochs=500)

eval_model=classifier.evaluate(x_test, y_test)
```


## PCA - Principal Component Analysis
[Source Paper](https://arxiv.org/pdf/1404.1100.pdf), [Video 1](https://www.youtube.com/watch?v=HMOI_lkzW08), [Video 2](https://youtu.be/kApPBm1YsqU) </br>
[Visualisation Notebook](https://github.com/mGalarnyk/Python_Tutorials/blob/master/Sklearn/PCA/PCA_Data_Visualization_Iris_Dataset_Blog.ipynb), [Dimensionality Reduction Notebook](https://github.com/mGalarnyk/Python_Tutorials/blob/master/Sklearn/PCA/PCA_to_Speed-up_Machine_Learning_Algorithms.ipynb)</br>
*PCA provides a roadmap for how to reduce a complex data set to a lower dimension to reveal the sometimes hidden, simplified structures that often underlie it.*
- PCA is intimately related to Singular Value Decomposition(SVD)
- The goal of principal component analysis is to identify the most meaningful basis to re-express a data set.

### PCA for data visualisation
```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Scale / Standardize the data
x = StandardScaler().fit_transform(x)

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])

finalDf = pd.concat([principalDf, df[['target']]], axis = 1)

# Visualisation
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 Component PCA', fontsize = 20)


targets = ['label 1', 'label 2', 'label 3'  ]
colors = ['r', 'g', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['target'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()

# Will tell us how much information is attributed to each of the principal components
# This should be ideally >85% to ensure significant information is not lost
pca.explained_variance_ratio_
```
### PCA for dimensionality reduction
```python
# Scale data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
train_img = scaler.fit_transform(train_img)
test_img = scaler.transform(test_img)

from sklearn.decomposition import PCA
# This creates an PCA instance which find the minimum number of components
# that will retain 95% of the variance/information from given dataset
pca = PCA(.95)

# Fit only on train
pca.fit(train_img)

# Transform both train and test
train_img = pca.transform(train_img)
test_img = pca.transform(test_img)

# Find the number of principal components selected
pca.n_components_
```
## SVM - Support Vector Machines
[Video](https://www.youtube.com/watch?v=efR1C6CvhmE), [Article](https://datafai.com/2018/08/10/support-vector-machine-svm/)</br>
*The objective of a SVM is to find the optimal separating hyperplane which maximises the separation between classes.*
If the original data is not linearly separatable then a *kernel trick* or *kernel transformation* is made to convert the data into a higher dimension to make it linearly separatable.
*If multiple hyperplanes are available, the best one is the one that maximizes the distance between the classes.*
If an SVM is used for classification, it’s called **Support Vector Classifier (SVC)**. Similarly, for regression it’s called **Support Vector Regressor (SVR)**.</br>
Two kernels are generally used:
- [The Polynomial Kernel](https://www.youtube.com/watch?v=Toet3EiSFcM) : Increases the low dimensional data to a **relatively higher dimension**
- [The Radial Basis Kernel (RBF)](https://www.youtube.com/watch?v=Qc5IyLW_hns) : Increases the low dimensional data to an **infinite dimension**</br>
Suitable use cases:
- Sparse data
- High Dimensional data
- Text Classification
- Data is nonlinear
- Image classification
- Data has complex patterns
### SVC - Suppor Vector Classifier
[docs](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
```python
from sklearn.svm import SVC
SVM = SVC(kernel='rbf', random_state=0, gamma=.10, C=1.0)
SVM.fit(X_train_standard, y_train)

print('Accuracy of our SVM model on the training data is {:.2f} out of 1'.format(SVM.score(X_train_standard, y_train)))
print('Accuracy of our SVM model on the test data is {:.2f} out of 1'.format(SVM.score(X_test_standard, y_test)))
```
### SVR - Support Vector Regressor
[docs](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html)
```python
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X, y)

y_pred = regressor.predict(x_test)
```
## Bias & Variance
[video](https://www.youtube.com/watch?v=EuBBz3bI-aA) </br>
*In ML, the ideal algorithm has **low bias** and can accurately model the true relationship and it has **low variability**, by producing consistent predictions across different datasets.*
Commonly used methods for finding the correct model with acceptably low bias and variance are:
- Regularisation
- Boosting
- Bagging













