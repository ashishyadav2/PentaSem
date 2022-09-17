import pandas as pd
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression

data = pd.read_csv("train.csv")
data.head()

data.info()

data_sub = data[['title', 'label']]
data_sub.head()

data_sub= data_sub.dropna()
data_sub= data_sub.reset_index()
data_sub.info()

plt.figure(figsize=(8,8))
sns.countplot(x='label', data=data_sub)
plt.xlabel('Classifier Real or Fake')
plt.ylabel('Count')
plt.show()

data_subset = data_sub.drop('label', axis=1)
data_subset.head()

stemmer = PorterStemmer() 

corpus=[]
for i in range(len(data_subset)):
    review= re.sub('[^A-Za-z]',' ',data_subset['title'][i])
    review= review.lower()
    review= review.split() #get list of words
    review= [stemmer.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review= ' '.join(review)
    corpus.append(review)

corpus[0]

cv = CountVectorizer(ngram_range =(2,2), max_features = 20000)
X = cv.fit_transform(corpus).toarray() # matrix creation- words as columns, sentences as rows

X

X_train, X_test, y_train, y_test = train_test_split(X, data_sub['label'], test_size =0.25, random_state =0)

lr_classifier = LogisticRegression()
lr_classifier.fit(X_train, y_train)

lr_y_pred = lr_classifier.predict(X_test)
confusion_mat = confusion_matrix(y_test, lr_y_pred)
confusion_mat

accuracy= accuracy_score(y_test, lr_y_pred)
print("Accuracy of Logistic Regression on Count Vectorizer data",accuracy*100)

tfidf = TfidfVectorizer(ngram_range =(2,2), max_features = 20000)
X_tf = tfidf.fit_transform(corpus).toarray() # matrix creation- words as columns, sentences as rows

X_tf

X_train, X_test, y_train, y_test = train_test_split(X_tf, data_sub['label'], test_size =0.25, random_state =0)

lr_classifier = LogisticRegression()
lr_classifier.fit(X_train, y_train)

lr_y_pred = lr_classifier.predict(X_test)
confusion_mat = confusion_matrix(y_test, lr_y_pred)
confusion_mat

accuracy= accuracy_score(y_test, lr_y_pred)
print("Accuracy of Logistic Regression on TfIdf Vectorizer data",accuracy*100)