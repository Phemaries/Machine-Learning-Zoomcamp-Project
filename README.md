# Machine Learning for Monkey Pox Prediction

This repository contains EDA and prediction of positive/negative cases of Monkeypox disease in patients using various classification models to get the optimum accuracy for prediction.

## Why this Project?

I am particularly motivated to find ways where severe health cases can be lessened and one of the current health challenges facing my country is Monkey Pox disease. Monkeypox (MPX) is endemic in Nigeria, but it was first reported in Adamawa state, North-Eastern Nigeria, in January 2022. There are currently 172 cases of MPX in Nigeria, with four reported deaths, and Adamawa has the second-highest case count.

## Data
The data used for this study is collected through published study: Clinical features and novel presentations of human monkeypox in a central London centre during the 2022 outbreak: descriptive case series, which is sourced from Kaggle. Check the dataset [here](https://www.kaggle.com/datasets/muhammad4hmed/monkeypox-patients-dataset) . The dataset is a CSV file with a record of 25,000 Patients with their corresponding features and a target variable indicating if the patient has monkeypox or not.

### Data Features

| Feature       | Description           | 
| ------------- |:-------------:| 
| Patient_ID      | Patients' Unique ID| 
| Systemic Illness     | Types of Illness such as Fever, Swollen Lymph Nodes, Muscle Aches and Pain     |   
| Rectal Pain |Do they have Rectal Pain (True or False)      |  
| Sore Throat |Do they have Sore Throat (True or False)     | 
| Penile Oedema |Do they have Penile Oedema (True or False)    | 
| Oral Lesions |Do they have Oral Lesions (True or False)      | 
| Solitary Lesion |Do they have Solitary Lesion (True or False)    | 
| HIV Infection |Do they have HIV Infection (True or False)      | 
| Sexually Transmitted Infection |Do they have any Sexually Transmitted Infection  (True or False)    | 
| Monkeypox |Do they have MonkeyPox (Positive) or not (Negative)   | 


### Exploratory Data Analysis (EDA)
An extensive EDA was carried out on the dataset. Data had no null values and explored each feature against the target variable 'Monkeypox' using data visualization and other metrics. This also provided some basic answers to providing a best fit for prediction.

### Model Training.
Multiple models were trained and compared for the dataset (60% training/ 20% validation / 20% testing). Models such as Logistic Regression, RandomForestClassifier, DecisionTreeClassifier, and RidgeClassifier with optimized parameters were selected to get the best metrics based on ##accuracy, ##confusion matrix, ##precision, ##recall, and ##f1_score. 

### Finetuning the models
All selected models were fine-tuned and the `Precison_Recall` curve was plotted for all the models. The Random Forest Classifier performed best.

### Conclusion
Upon hypermetric tuning for RandomForest the model `RandomForestClassifier(n_estimators=120, random_state=42, n_jobs=-1, max_depth = 5)` has an accuracy of 70.22%. Run on `train.py` file

### Model and Deployment to Flask
* The best model is saved into `model.pkl` with the `dv` and `modelrf1 features.
* waitress-serve --listen=0.0.0.0:9696 predict:app
* Create a virtual environment using: python -m venv env
* In project directory run `venv\Scripts\activate`
* Install all packages `pip install [packages]`
* `pip freeze > requirements.txt` to create `requirement.txt` file
* run `python predict_test.py`


### Future Scope
* Containerization and Deployment to Cloud
