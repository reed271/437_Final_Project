# Vaccine Project Documentation

## Data source
https://www.kaggle.com/datasets/prox37/twitter-multilabel-classification-dataset

Upload this to the first colab cell. This is a list with the columns [tweet text, tweet class]. 

## Data preprocessing

To first process the data, we first get a network of word positions via nltk, and also get the stop words from nltk. The data is converted into a numpy array. Each line is grabbed, split into words, and then checked for symbols and stop words.

### Data --> split --> stop word/symbol filter --> filtered data

If it is not a symbol or stop word, the preprocessing calculates a lemmatized version of the word. It then adds the lemmatized word to a string, with a space, and then once the sentence is processed, it is added to the processed data list.


### Filtered data --> lemmatizer --> concatenation --> append

One the data is filtered, the classes are extracted. Since it is multi-output, we get the present classes for each data point, setting 1 if it exists and 0 if it doesn't.

## Feature extraction

Features are generated via the generate_data function. It takes in the X, Y vectors, and first splits them into train/test. It then generates a dictionary of different types of data. The dictionary keys are data type name. The values of the dict are lists of test/train data. These lists correspond to the possible permutations within a feature type - for example, it will count various CountVectorized test/train data, using different ngram ranges. It creates Count and TF-IDF vectors using ngram ranges of [(1,2), (1,3), (2,3)]. It also used a min_df value of [2,5,8,10].

### X, Y data --> split --> vectorizer hyperparam looping --> vectorizer construction

### vectorizer --> vectorized data --> append

## Model training

Since this is a multi-output model, instead of training for multi-classification, binary models are instead created for each class. Each model is then trained and evaluated and the predictions from each class can be added together for the multi-output problem. After training, the models are evaluated on the test set, and the average scores are returned

### Models/classes --> loop through classes --> train model on class --> append model

### trained models --> test evaluation --> average scores

## Model Creation

A variety of different models were created. The following is a list of the models used for later selection, with tested hyperparams (the best were kept to speed up the training process):

Linear SVM with (C = 1.2, loss = 'hinge', class_weight = 'balanced')

    Tried:
    - alpha = 0.2, 0.5, 0.8, 1
    - force_alpha = false, true

Multinomial NB with (alpha = .4, force_alpha = True, fit_prior = True)

    Tried:
    - alpha = 0.2, 0.5, 0.8, 1
    - force_alpha = false, true
Bernoulli NB, BernoulliNB(alpha = .8, force_alpha = True, fit_prior = True)

    Tried:
    - alpha = 0.2, 0.5, 0.8, 1
    - force_alpha = false, true

Random Forest, RandomForestClassifier(n_estimators=150)
    
    Tried:
    - n_estimators = 100, 150, 200

GradientBoostingClassifier()

## Model scoring

The model is evaluated using 2 functions - a given score function, which by default is F1, and an accuracy score. The evaluation loops through each class/model, and calculates the F1 score for that binary classifier. It then does this for each class, averaging the scores at the end. It then plots a bar chart of the scores for each class, and returns the average scores. 

### Models/classes + data --> evaluate --> average --> plot --> return

## Model selection

We went through every created model to find the best one. It first enumerates through each type of data, and each given train/test permutation of the vectorizer. Once it has the train/test data, it goes through every created model, and evaluates it. The best scores are saved, along with the model and data name. Once the data is looped through, it returns the best model along with the scores.

### Created models + data --> data type enumeration --> feature variation enumeration

### features --> model type enumeration --> model hyperparam enumeration

### model --> score for accuracy/f1 --> score averaging --> plotting --> find max score