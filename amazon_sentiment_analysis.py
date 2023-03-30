#####################################################
# Sentiment Analysis for Amazon Review
#####################################################


#####################################################
# Business Problem
#####################################################

# Kozmos, which makes home textile and daily clothing-oriented productions that realize sales through Amazon, makes,
# by analyzing the reviews received for its products and improving its features according to the complaints it receives,
# its sales it aims to increase. In line with this goal, the comments will be labeled by conducting a sentiment analysis
# and a classification model will be created with the labeled data.

#####################################################
# The Data Set Story
#####################################################

# The dataset includes the comments made for a specific product group, the comment title,
# the number of stars and the comment made it consists of variables that indicate how many people find it useful.

# Star: The number of stars awarded to the product
# HelpFul: The number of people who found the comment useful
# Title: The title given to the content of the comment, a short comment
# Review: Comments on the product

#####################################################
# Tasks
#####################################################


from warnings import filterwarnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate
from sklearn.preprocessing import LabelEncoder
from textblob import Word, TextBlob
from wordcloud import WordCloud

#####################################################
# Text Pre-Processing
#####################################################

# Step1: Read amazon.xlsx data

df = pd.read_excel("amazon.xlsx")
df.head()
df.info()

# Step2: On the review variable ;
       # a. Translate all letters to lowercase.
       # b. Remove the punctuation marks.
       # c. Take out the numerical expressions in the comments.
       # d. Remove the words (stopwords) that do not contain information from the data.
       # e. subtract less than 1000 words from the data.
       # f. Apply the lemmatization process.

# Normalizing Case Folding

df['Review'] = df['Review'].str.lower()

# Punctuations

df['Review'] = df['Review'].str.replace('[^\w\s]', '')

# Numbers

df['Review'] = df['Review'].str.replace('\d', '')

# Stopwords

# import nltk
# nltk.download('stopwords')

sw = stopwords.words('english')

df['Review'] = df['Review'].apply(lambda x: " ".join(x for x in str(x).split() if x not in sw))

# Rarewords

drops = pd.Series(" ".join(df["Review"]).split()).value_counts()[-1000:]
df["Review"] = df["Review"].apply(lambda x: " ".join(x for x in x.split() if x not in drops))

# Lemmatization

df['Review'] = df['Review'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
df["Review"].head()

#####################################################
# Text Visualization
###################################################

# Step1: For the barplot visualization process;
       # a. Calculate the frequencies of the words contained in the "Review" variable, save them as tf.
       # b. rename the columns of the tf dataframe as: "words", "tf"
       # c. the visualization process with barplot by filtering the value of the variable "tf" according to those
       # that are more than 500 complete it

tf = df["Review"].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()
tf.columns = ["words","tf"]

tf[tf["tf"] > 500].plot.bar(x="words", y="tf")
plt.show(block= True)

# Step2: For the WordCloud visualization process;
       # a. Save all the words contained in the "review" variable as a string in the name "text".
       # b. Define and save your template shape using WordCloud.
       # c. Generate the wordcloud you saved with the string you created in the first step.
       # d. Complete the visualization steps. (figure, imshow, axis, show)

text = " ".join(i for i in df.Review)
wordcloud= WordCloud(max_font_size=50,
                     max_words=100,
                     background_color="white").generate(text)

plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show(block=True)

wordcloud.to_file("amzwordcloud.png")

#####################################################
# Sentiment Analysis
###################################################

# Step 1: Create the SentimentIntensityAnalyzer object defined in the NLTK package in Python.

sia = SentimentIntensityAnalyzer()

# Step 2: Examine the polarity scores with the Sentiment intensity analyzer object;
      # a. Calculate polarity_scores() for the first 10 observations of the "Review" variable.
      # b. For the first 10 observations examined, please observe again by filtering according to the compund scores.
      # c. if the compound scores are greater than 0 for 10 observations, update them as "neg" if not "pos".
      # d.By assigning pos-neg for all observations in the variable "Review" to the dataframe as a new variable add

df["Review"][0:10].apply(lambda x: sia.polarity_scores(x))

df["Review"][0:10].apply(lambda x: sia.polarity_scores(x)["compound"])

df["polarity_score"] = df["Review"].apply(lambda x: sia.polarity_scores(x)["compound"])

df["Review"][0:10].apply(lambda x: "pos" if sia.polarity_scores(x)["compound"] > 0 else "neg")

df["sentiment_label"] = df["Review"].apply(lambda x: "pos" if sia.polarity_scores(x)["compound"] > 0 else "neg")

df["sentiment_label"].value_counts()
df.groupby("sentiment_label")["Star"].mean()

#####################################################
# Preparation for Machine Learning
###################################################

# Step 1: Separate the data as a train test by determining our dependent and independent variables

# Test-Train

from sklearn.model_selection import train_test_split

train_x, test_x, train_y, test_y = train_test_split(df["Review"],
                                                    df["sentiment_label"])


# Step 2: In order to provide the data to the machine learning model, we need to convert the representation shapes to digital;
       # a. Create an object using the TfidfVectorizer.
       # b. Please fit the object we have created using our train data that we have previously allocated.
       # c. Apply the transformation process and save the vector we have created to the train and test data

from sklearn.feature_extraction.text import TfidfVectorizer

tf_idf_word_vect = TfidfVectorizer().fit(train_x)
x_train_tf_idf_word = tf_idf_word_vect.transform(train_x)
x_test_tf_idf_word = tf_idf_word_vect.transform(test_x)

#####################################################
# Modeling (Logistic Regression)
###################################################

# Step 1: Set up the logistic regression model and fit it with the train data.

log_model = LogisticRegression().fit(x_train_tf_idf_word, train_y)


# Step 2: Perform prediction operations with the model you have installed;
      # a. Record the test data by estimating it with the Predict function.
      # b. report and observe your forecast results with classification_report.
      # c. calculate the average accuracy value using the cross validation function.

y_pred = log_model.predict(x_test_tf_idf_word)

from sklearn.metrics import classification_report
print(classification_report(y_pred, test_y))

cross_val_score(log_model, x_test_tf_idf_word, test_y, cv=5).mean()
# 0.8688

# Step 3: Selecting the rating from the comments found in the data and asking the model;
      # a. with the sample function, select a sample from the "Review" variable and assign it to a new value.
      # b. Vectorize the sample you have obtained with the CountVectorizer so that the model can predict.
      # c. Save the sample you have vectorized by doing the fit and transform operations.
      # d. Record the prediction result by giving the sample to the model you have set up.
      # e. Print the sample and the forecast result on the screen

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
random_review = pd.Series(df["Review"].sample(1).values)
new_comment = CountVectorizer().fit(train_x).transform(random_review)
pred = log_model.predict(new_comment)
print(f'Review: {random_review[0]} \n Prediction:{pred} ')

#####################################################
# Modeling (Random Forest)
###################################################

#Step 1: Observation of prediction results with Random Forest model;
      #a. Install and fit the Random Forest Classifier model.
      #b. Calculate the average accuracy value using the cross validation function.
      #c. Compare the results with the logistic regression model

rf_model = RandomForestClassifier().fit(x_train_tf_idf_word, train_y)
cross_val_score(rf_model, x_test_tf_idf_word, test_y, cv=5, n_jobs=1).mean()
#0.8994
