## CONDITIONS FAVOURABLE PLANTING DIFFERENT _CROPS_

### About Dataset

---

- The dataset [Agriculture Data](./Agriculture%20data.xlsx) contains 2200 rows and 8 columns.
- It provides favourable conditions that favour growth of some plants, the conditions include: Nitrogen(**N**), Phosphorus(**P**), Potassium(**K**), **temperature, humidity, ph and rainfall**.
- The plants predicted are:
    `['rice', 'maize', 'chickpea', 'kidneybeans', 'pigeonpeas', 'mothbeans', 'mungbean', 'blackgram', 'lentil', 'pomegranate', 'banana', mango', 'grapes', 'watermelon', 'muskmelon', 'apple', 'orange', 'papaya', 'coconut', 'cotton', 'jute', 'coffee']`

---

### About the Model

---

- The model takes input of the following items: `Nitrogen(**N**), Phosphorus(**P**), Potassium(**K**), **temperature, humidity, ph and rainfall**`.
- From that input, it returns the prediction object and the crop which has been predicted.
- All inputs are saved in the database for retraining the model and the result to is saved to track the performance of the model over time.
- The model takes floating point values to predict the crop which is favourable for the given conditions.

---

### Technical details about the model

- The model is based on `LogisticRegression` and predicts 22 different classes.
- Model accuracy after Cross Validation and model selection is `96.6%`.
- The model MinMaxScales all the inputs using `sklearn.preprocessing.MinMaxScaler` to have the values withing a given range for ease in training the model and creating predictions.

### To Use this repo:

---