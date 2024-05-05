# MLNetTraffic
This project aims to analyze and classify a real network traffic dataset to detect malicious/benign traffic records. It compares and tunes the performance of several Machine Learning algorithms to maintain the highest accuracy and lowest False Positive/Negative rates.

![Machine Learning in Cyber Security](https://media.geeksforgeeks.org/wp-content/cdn-uploads/20201210221120/Top-5-Applications-of-Machine-Learning-in-Cyber-Security1.jpg)
|        |
| :---:                        |
[![Video Explanation](https://img.youtube.com/vi/n794BuOW8QY/0.jpg)](https://www.youtube.com/watch?v=n794BuOW8QY)

Explanation of the model by RANJI RAJ


## Data Set 
The dataset used in this demo is: [CTU-IoT-Malware](https://mcfp.felk.cvut.cz/publicDatasets/IoT-23-Dataset/IndividualScenarios/CTU-IoT-Malware-Capture-34-1/bro/).
- this dataset for your research, please reference it as “Sebastian Garcia, Agustin Parmisano, & Maria Jose Erquiaga. (2020). IoT-23: A labeled dataset with malicious and benign IoT network traffic (Version 1.0.0) [Data set]. Zenodo. http://doi.org/10.5281/zenodo.4743746”

## Data Classification Details
The project is implemented in four distinct steps simulating the essential data processing and analysis phases. <br/>
- Each step is represented in a corresponding notebook inside [notebooks](notebooks).
- Intermediary data files are stored inside the [data](data) path.
- Trained models are stored inside [models](models).

### Initial Data Cleaning
> Corresponding notebook:  [cleaning.ipynb](https://github.com/michredteam/mlnettraffic/blob/main/notebooks/1-cleaning.ipynb)

Implemented data exploration and cleaning tasks:
1. Loading the raw dataset file into pandas DataFrame.
2. Exploring dataset summary and statistics.
3. Fixing combined columns.
4. Dropping irrelevant columns.
5. Fixing unset values and validating data types.
6. Checking the cleaned version of the dataset.
7. Storing the cleaned dataset to a csv file.

### Data Processing
> Corresponding notebook:  [preprocessing.ipynb](https://github.com/michredteam/mlnettraffic/blob/main/notebooks/2-preprocessing.ipynb)

Implemented data processing and transformation tasks:
1. Loading dataset file into pandas DataFrame.
2. Exploring dataset summary and statistics.
3. Analyzing the target attribute.
4. Encoding the target attribute using [LabelEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html).
5. Handling outliers using [IQR (Inter-quartile Range)](https://en.wikipedia.org/wiki/Interquartile_range).
6. Encoding IP addresses.
7. Handling missing values:
    1. Impute missing categorical features using [KNeighborsClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html).
    2. Impute missing numerical features using [KNNImputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html).
8. Scaling numerical attributes using [MinMaxScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html).
9. Encoding categorical features: handling rare values and applying [One-Hot Encoding](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html).
10. Checking the processed dataset and storing it to a csv file.

### Model Training
> Corresponding notebook:  [training.ipynb](https://github.com/michredteam/mlnettraffic/blob/main/notebooks/3-training.ipynb)

Trained and analyzed classification models:
1. Naive Bayes: [ComplementNB](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.ComplementNB.html)
2. Decision Tree: [DecisionTreeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
3. Logistic Regression: [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)    
4. Random Forest: [RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
5. Support Vector Classifier: [SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC)
6. K-Nearest Neighbors: [KNeighborsClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
7. XGBoost: [XGBClassifier](https://xgboost.readthedocs.io/en/stable/index.html#)

Evaluation method: 
- Cross-Validation Technique: [Stratified K-Folds Cross-Validator](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html)
- Folds number: 5
- Shuffled: Enabled

Results were analyzed and compared for each considered model.

### Model Tuning
> Corresponding notebook:  [tuning.ipynb](https://github.com/michredteam/mlnettraffic/blob/main/notebooks/4-tuning.ipynb)

Model tuning details:
- Tuned model: Support Vector Classifier - [SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC)
- Tuning method: [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
- Results were analyzed before/after tuning.



 
