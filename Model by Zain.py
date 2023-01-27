print("hello")
import re
import pandas as pd
import numpy as np
import seaborn as sn
import string
import matplotlib.pyplot as plt 
import nltk
from nltk import TweetTokenizer
from nltk import PorterStemmer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, accuracy_score

from prettytable import PrettyTable
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

train = pd.read_csv("train.csv", header = None)
train.columns = ["Comment", "Polarity"]
train_original = train.copy()
train

test = pd.read_csv("test.csv", header = None)
test.columns = ["Comment", "Polarity"]
test_original = test.copy()
test

# Understand Train Data

print("\n\nAttributes in Sample Data:")
print("==========================\n")

print(train.columns)

print("\n\nNumber of Instances in Train Data:",train["Polarity"].count())
print("========================================\n")

# Understand Test Data

print("\n\nAttributes in Test Data:")
print("==========================\n")

print(train.columns)

print("\n\nNumber of Instances in Test Data:",test["Polarity"].count())
print("========================================\n")

print("\n\nTotal Number of Instances in Data:",train["Polarity"].count() + test["Polarity"].count())
print("========================================\n")

combined_data = train.append(test,ignore_index=True,sort=True)
print(combined_data)

def remove_pattern(comment, pattern):
#     r = re.findall(pattern, comment)
#     for i in r:
    comment = re.sub(pattern,"", comment)
    return comment
combined_data["Tidy_Comment"] = np.vectorize(remove_pattern)(combined_data["Comment"], "@[\w]*")
print(combined_data)

combined_data["Tidy_Comment"] = combined_data["Tidy_Comment"].str.replace("[^a-zA-Z#\s]", "")
print(combined_data)

combined_data['Tidy_Comment'] = combined_data['Tidy_Comment'].apply(
    lambda x: ' '.join([w for w in x.split() if len(w) > 3]))
print(combined_data)

Tokenizer = TweetTokenizer()
combined_data["Tidy_Comment"] = combined_data['Tidy_Comment'].apply(lambda x: Tokenizer.tokenize(str(x)))
print(combined_data)

ps = PorterStemmer()
combined_data['Tidy_Comment'] = combined_data['Tidy_Comment'].apply(
    lambda comment: [ps.stem(letter) for letter in comment])
print(combined_data)

for i in range(len(combined_data["Tidy_Comment"])):
    combined_data["Tidy_Comment"][i] = ' '.join(combined_data["Tidy_Comment"][i])
print(combined_data)

# Train the Label Encoder

''' 
*------------------ TRAIN_LABEL_ENCODER --------------------*
|        Function: Fit()                                    |
|              Purpose: Fit or Train the Label Encoder      |
|        Arguments:                                         |
|               Labels: Target Values                       |
|        Return:                                            |
|               Instance: Returns an instance of self       |
*-----------------------------------------------------------*
''' 

# Label
polarity = pd.DataFrame({"Polarity": ["Happy", "Sad"]})
# Initializ the label encoder
polarity_label_encoder = LabelEncoder()
# Train the Label Encoder
polarity_label_encoder.fit(np.ravel(polarity))

train_encoded_Polarity = train_original.copy()
original_train_data = train_original.copy()

# Label Encoding of the Output

''' 
*------------------ LABEL_ENCODE_OUTPUT --------------------*
|        Function: Transform()                              |
|              Purpose: Transform Input (Categorical)       |
|                       into Numerical Representation       |
|        Arguments:                                         |
|              Attribute: Target values                     |
|        Return:                                            |
|              Attribute: Numerical Representation          |
*-----------------------------------------------------------*
'''


# Transform Output of into Numerical Representation

print("\n\nOutput Attribute After Label Encoding:")
print("========================================\n")
train["Encoded_Polarity"] = polarity_label_encoder.transform(train['Polarity'])
print(train[["Polarity", "Encoded_Polarity"]])

# Print Original and Encoded Ouput Sample Data

train_encoded_polarity = train
print("\n\nOriginal Train Data:")
print("=====================\n")
#pd.set_option("display.max_rows", None, "display.max_columns", None)
print(original_train_data)
print("\n\nTrain Data after Label Encoding of Output:")
print("===========================================\n")
#pd.set_option("display.max_rows", None, "display.max_columns", None)
print(train_encoded_polarity)

# Save the Transformed Features into CSV File 

# sample_data_encoded_output.to_csv(r'sample-data-encoded-output.csv', index = False, header = True)

# Train the Label Encoder

''' 
*------------------ TRAIN_LABEL_ENCODER --------------------*
|        Function: Fit()                                    |
|              Purpose: Fit or Train the Label Encoder      |
|        Arguments:                                         |
|               Labels: Target Values                       |
|        Return:                                            |
|               Instance: Returns an instance of self       |
*-----------------------------------------------------------*
''' 

# Label
polarity = pd.DataFrame({"Polarity": ["Happy", "Sad"]})
# Initializ the label encoder
polarity_label_encoder = LabelEncoder()
# Train the Label Encoder
polarity_label_encoder.fit(np.ravel(polarity))

test_encoded_polarity = test_original.copy()
original_test_data = test_original.copy()

# Label Encoding of the Output

''' 
*------------------ LABEL_ENCODE_OUTPUT --------------------*
|        Function: Transform()                              |
|              Purpose: Transform Input (Categorical)       |
|                       into Numerical Representation       |
|        Arguments:                                         |
|              Attribute: Target values                     |
|        Return:                                            |
|              Attribute: Numerical Representation          |
*-----------------------------------------------------------*
'''

# Transform Output of into Numerical Representation

print("\n\nOutput Attribute After Label Encoding:")
print("========================================\n")
test["Encoded_Polarity"] = polarity_label_encoder.transform(test['Polarity'])
print(test[["Polarity", "Encoded_Polarity"]])

# Print Original and Encoded Ouput Sample Data

test_encoded_polarity = test
print("\n\nOriginal Test Data:")
print("=====================\n")
#pd.set_option("display.max_rows", None, "display.max_columns", None)
print(original_test_data)
print("\n\nTest Data after Label Encoding of Output:")
print("===========================================\n")
#pd.set_option("display.max_rows", None, "display.max_columns", None)
print(test_encoded_polarity)

# Save the Transformed Features into CSV File 

# sample_data_encoded_output.to_csv(r'sample-data-encoded-output.csv', index = False, header = True)

# Train the Label Encoder

''' 
*------------------ TRAIN_LABEL_ENCODER --------------------*
|        Function: Fit()                                    |
|              Purpose: Fit or Train the Label Encoder      |
|        Arguments:                                         |
|               Labels: Target Values                       |
|        Return:                                            |
|               Instance: Returns an instance of self       |
*-----------------------------------------------------------*
''' 

# Label
polarity = pd.DataFrame({"Polarity": ["Happy", "Sad"]})
# Initializ the label encoder
polarity_label_encoder = LabelEncoder()
# Train the Label Encoder
polarity_label_encoder.fit(np.ravel(polarity))

combined_data_encoded_polarity = combined_data.copy()
original_combined_data = combined_data.copy()

# Label Encoding of the Output

''' 
*------------------ LABEL_ENCODE_OUTPUT --------------------*
|        Function: Transform()                              |
|              Purpose: Transform Input (Categorical)       |
|                       into Numerical Representation       |
|        Arguments:                                         |
|              Attribute: Target values                     |
|        Return:                                            |
|              Attribute: Numerical Representation          |
*-----------------------------------------------------------*
'''

# Transform Output of into Numerical Representation

print("\n\nOutput Attribute After Label Encoding:")
print("========================================\n")
combined_data["Encoded_Polarity"] = polarity_label_encoder.transform(combined_data['Polarity'])
print(combined_data[["Polarity", "Encoded_Polarity"]])

# Print Original and Encoded Ouput Sample Data

combined_data_encoded_polarity = combined_data
print("\n\nOriginal Combined Data:")
print("=====================\n")
#pd.set_option("display.max_rows", None, "display.max_columns", None)
print(original_combined_data)
print("\n\nCombined Data after Label Encoding of Output:")
print("===========================================\n")
#pd.set_option("display.max_rows", None, "display.max_columns", None)
print(combined_data_encoded_polarity)

# Save the Transformed Features into CSV File 

# sample_data_encoded_output.to_csv(r'sample-data-encoded-output.csv', index = False, header = True)

count_vectorizer = CountVectorizer(max_df=0.90,
                                   min_df=2,
                                   max_features=1000,
                                   stop_words='english')

# Step 3. Create the Bag-of-Words Model
bag_of_words = count_vectorizer.fit_transform(combined_data['Tidy_Comment'])

# Show the Bag-of-Words Model as a pandas DataFrame
feature_names = count_vectorizer.get_feature_names_out()
pd.DataFrame(bag_of_words.toarray(), columns=feature_names)

tfidf_vectorizer = TfidfVectorizer(max_df=0.90, min_df=2, stop_words='english')
values = tfidf_vectorizer.fit_transform(combined_data['Tidy_Comment'])

# Show the Model as a pandas DataFrame
feature_names = tfidf_vectorizer.get_feature_names_out()
pd.DataFrame(values.toarray(), columns = feature_names)
print(feature_names)

train_bow = bag_of_words
train_bow.todense()
print(train_bow.todense())

train_tfidf_matrix = values
train_tfidf_matrix.todense()

x_train_bow, x_valid_bow, y_train_bow, y_valid_bow = train_test_split(train_bow,combined_data['Encoded_Polarity'],test_size=0.3,random_state=2)
x_valid_bow

# Train the Support Vector Classifier

''' 
*--------------- TRAIN_SUPPORT_VECTOR_CLASSIFIER ------------------*
|       Function: DecisionTreeClassifier()                         |
|           Purpose: Train the Algorithm on Training Data          |
|       Arguments:                                                 |
|           Training Data: Provide Training Data to the Model      |
|       Return:                                                    |
|           Parameter: Model return the Training Parameters        |
*------------------------------------------------------------------*
'''

print("\n\nTraining the Decision Tree algorithm on Training Data")
print("========================================================\n")
print("\nParameters and their values:")
print("============================\n")
from sklearn.tree import DecisionTreeClassifier
model_bow = DecisionTreeClassifier(criterion='entropy', random_state=1)
model_bow.fit(x_train_bow,y_train_bow)
print(model_bow.fit(x_train_bow,y_train_bow))


# Save the Trained Model

''' 
*--------------------- SAVE_THE_TRAINED_MODEL ---------------------*
|        Function: dump()                                          |
|             Purpose: Save the Trained Model on your Hard Disk    |
|        Arguments:                                                |
|             Model: Model Objects                                 |
|        Return:                                                   |
|             File: Trained Model will be Saved on Hard Disk       |
*------------------------------------------------------------------* 
'''

# Save the Model in a Pkl File
import pickle
pickle.dump(model_bow, open('decision_tree_bow_trained_model.pkl', 'wb'))


# Load the Saved Model

''' 
*------------------- LOAD_SAVED_MODEL --------------------------*
|         Function: load()                                      |
|               Purpose: Method to Load Previously Saved Model  |
|         Arguments:                                            |
|               Model: Trained Model                            |
|         Return:                                               |
|               File: Saved Model will be Loaded in Memory      |
*---------------------------------------------------------------*
'''

# Load the Saved Model

model = pickle.load(open('decision_tree_bow_trained_model.pkl', 'rb'))


dct_bow = model.predict_proba(x_valid_bow)
print(dct_bow)


# if prediction is greater than or equal to 0.3 than 1 else 0
# Where 0 is for positive sentiment tweets and 1 for negative sentiment tweets
dct_bow=dct_bow[:,1]>=0.3


# Calculate the Accuracy Score

''' 
/*------------------------ CALCULATE_ACCURACY_SCORE -------------------*
|          Function: accuracy_score()                                  |
|                Purpose: Evaluate the algorithm on Testing data       |
|          Arguments:                                                  |
|                Prediction: Predicted values                          |
|                Label: Actual values                                  |
|          Return:                                                     |
|                Accuracy: Accuracy Score                              |
*----------------------------------------------------------------------*
'''
# converting the results to integer type
dct_int_bow=dct_bow.astype(np.int)
print(y_valid_bow)
print(dct_int_bow)

# calculating accuracy score
dct_score_bow=accuracy_score(y_valid_bow,dct_int_bow)

print("\n\nAccuracy Score by using BOW Features to make Predictions:")
print("===========================================================\n")
print(dct_score_bow)


x_train_tfidf, x_valid_tfidf, y_train_tfidf, y_valid_tfidf = train_test_split(train_tfidf_matrix,combined_data['Encoded_Polarity'],test_size=0.3,random_state=17)


# Train the Support Vector Classifier

''' 
*--------------- TRAIN_SUPPORT_VECTOR_CLASSIFIER ------------------*
|       Function: DecisionTreeClassifier()                         |
|           Purpose: Train the Algorithm on Training Data          |
|       Arguments:                                                 |
|           Training Data: Provide Training Data to the Model      |
|       Return:                                                    |
|           Parameter: Model return the Training Parameters        |
*------------------------------------------------------------------*
'''

print("\n\nTraining the Decision Tree algorithm on Training Data")
print("========================================================\n")
print("\nParameters and their values:")
print("============================\n")
from sklearn.tree import DecisionTreeClassifier
model_bow = DecisionTreeClassifier(criterion='entropy', random_state=1)
model_bow.fit(x_train_tfidf,y_train_tfidf)

print(model_bow.fit(x_train_tfidf,y_train_tfidf))


# Save the Trained Model

''' 
*--------------------- SAVE_THE_TRAINED_MODEL ---------------------*
|        Function: dump()                                          |
|             Purpose: Save the Trained Model on your Hard Disk    |
|        Arguments:                                                |
|             Model: Model Objects                                 |
|        Return:                                                   |
|             File: Trained Model will be Saved on Hard Disk       |
*------------------------------------------------------------------* 
'''

# Save the Model in a Pkl File
import pickle
pickle.dump(model_bow, open('decision_tree_tfidf_trained_model.pkl', 'wb'))

# Load the Saved Model

''' 
*------------------- LOAD_SAVED_MODEL --------------------------*
|         Function: load()                                      |
|               Purpose: Method to Load Previously Saved Model  |
|         Arguments:                                            |
|               Model: Trained Model                            |
|         Return:                                               |
|               File: Saved Model will be Loaded in Memory      |
*---------------------------------------------------------------*
'''

# Load the Saved Model

model = pickle.load(open('decision_tree_tfidf_trained_model.pkl', 'rb'))

dct_tfidf = model.predict_proba(x_valid_tfidf)

print(dct_tfidf)


# if prediction is greater than or equal to 0.3 than 1 else 0
# Where 0 is for positive sentiment tweets and 1 for negative sentiment tweets
dct_tfidf=dct_tfidf[:,1]>=0.3

# Calculate the Accuracy Score

''' 
/*------------------------ CALCULATE_ACCURACY_SCORE -------------------*
|          Function: accuracy_score()                                  |
|                Purpose: Evaluate the algorithm on Testing data       |
|          Arguments:                                                  |
|                Prediction: Predicted values                          |
|                Label: Actual values                                  |
|          Return:                                                     |
|                Accuracy: Accuracy Score                              |
*----------------------------------------------------------------------*
'''
# converting the results to integer type
dct_int_tfidf=dct_tfidf.astype(np.int)
# calculating accuracy score
dct_score_tfidf=accuracy_score(y_valid_tfidf,dct_int_tfidf)

print("\n\nAccuracy Score by using TFIDF Features to make Predictions:")
print("===========================================================\n")
print(dct_score_tfidf)




#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Application Phase>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
# Take Input from User
''' 
---------------- TAKE_USER_INPUT ----------------
'''
tweet = str(input("\nPlease enter tweet here : "))
user_input = pd.DataFrame({"Comment": [tweet]})
print(user_input)


def remove_pattern(comment, pattern):
#     r = re.findall(pattern, comment)
#     for i in r:
    comment = re.sub(pattern,"", comment)
    return comment
user_input["Tidy_Comment"] = np.vectorize(remove_pattern)(user_input["Comment"], "@[\w]*")
print(user_input)

user_input["Tidy_Comment"] = user_input["Tidy_Comment"].str.replace("[^a-zA-Z#\s]", "")
print(user_input)

user_input['Tidy_Comment'] = user_input['Tidy_Comment'].apply(
    lambda x: ' '.join([w for w in x.split() if len(w) > 3]))
print(user_input)


Tokenizer = TweetTokenizer()
user_input["Tidy_Comment"] = user_input['Tidy_Comment'].apply(lambda x: Tokenizer.tokenize(str(x)))
print(user_input)

ps = PorterStemmer()
user_input['Tidy_Comment'] = user_input['Tidy_Comment'].apply(
    lambda comment: [ps.stem(letter) for letter in comment])
print(user_input)

for i in range(len(user_input["Tidy_Comment"])):
    user_input["Tidy_Comment"][i] = ' '.join(user_input["Tidy_Comment"][i])
print(user_input)


tfidf_vectorizer = TfidfVectorizer(max_features=2, stop_words='english')
values = tfidf_vectorizer.fit_transform(user_input['Tidy_Comment'])

# Show the Model as a pandas DataFrame
feature_names = tfidf_vectorizer.get_feature_names_out()
print(pd.DataFrame(values.toarray(), columns = feature_names))


user_input_tfidf_matrix = values
user_input_tfidf_matrix.todense()
print(user_input_tfidf_matrix.todense())

# Load the Saved Model

''' 
*------------------- LOAD_SAVED_MODEL --------------------------*
|         Function: load()                                      |
|               Purpose: Method to Load Previously Saved Model  |
|         Arguments:                                            |
|               Model: Trained Model                            |
|         Return:                                               |
|               File: Saved Model will be Loaded in Memory      |
*---------------------------------------------------------------*
'''

# Load the Saved Model

model = pickle.load(open('decision_tree_tfidf_trained_model.pkl', 'rb'))

#.......................................................output.....................................#
dct_tfidf = model.predict_proba(user_input_tfidf_matrix)
dct_tfidf=dct_tfidf[:,1]>=0.5
dct_int_tfidf=dct_tfidf.astype(np.int)
dct_int_tfidf
if dct_int_tfidf == 0:
    print("Prediction: Happy")
if dct_int_tfidf == 1:
    print("Prediction: Sad")