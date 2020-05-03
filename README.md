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
[Source Paper](https://arxiv.org/pdf/1404.1100.pdf), [Video](https://www.youtube.com/watch?v=HMOI_lkzW08) </br>
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















