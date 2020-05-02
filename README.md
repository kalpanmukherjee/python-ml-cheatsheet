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
