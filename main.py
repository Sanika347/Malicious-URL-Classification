from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
import pandas as pd
import warnings
import pickle
import re

warnings.filterwarnings("ignore", category=FutureWarning)

url_df = pd.read_csv('train_dataset.csv')
#url_df = pd.read_csv('Malicious URLs.csv')


def tokenizer(url):
    tokens = re.split('[/-]', url)

    for i in tokens:
        if i.find(".") >= 0:
            dot_split = i.split('.')

            if "com" in dot_split:
                dot_split.remove("com")
            if "www" in dot_split:
                dot_split.remove("www")

            tokens += dot_split

    tokens = list(filter(None, tokens))
    return tokens


train_df, test_df = train_test_split(url_df, test_size=0.3, random_state=42)

labels = train_df['Class']
test_labels = test_df['Class']

cVec = CountVectorizer(tokenizer=tokenizer)
count_X = cVec.fit_transform(train_df['URLs'])

tVec = TfidfVectorizer(tokenizer=tokenizer)
tfidf_X = tVec.fit_transform(train_df['URLs'])

print("- Count Vectorizer -")
test_count_X = cVec.transform(test_df['URLs'])

print("- TFIDF Vectorizer -")
test_tfidf_X = tVec.transform(test_df['URLs'])


lgs_tfidf = LogisticRegression(solver='lbfgs')
lgs_tfidf.fit(tfidf_X, labels)

score_lgs_tfidf = lgs_tfidf.score(test_tfidf_X, test_labels)
predictions_lgs_tfidf = lgs_tfidf.predict(test_tfidf_X)

lgs_count = LogisticRegression(solver='lbfgs')
lgs_count.fit(count_X, labels)

score_lgs_count = lgs_count.score(test_count_X, test_labels)
predictions_lgs_count = lgs_count.predict(test_count_X)


mnb_tfidf = MultinomialNB()
mnb_tfidf.fit(tfidf_X, labels)

score_mnb_tfidf = mnb_tfidf.score(test_tfidf_X, test_labels)
predictions_mnb_tfidf = mnb_tfidf.predict(test_tfidf_X)

mnb_count = MultinomialNB()
mnb_count.fit(count_X, labels)

score_mnb_count = mnb_count.score(test_count_X, test_labels)
predictions_mnb_count = mnb_count.predict(test_count_X)


svm_tfidf = SVC(kernel='linear', probability=True)
svm_tfidf.fit(tfidf_X, labels)

score_svm_tfidf = svm_tfidf.score(test_tfidf_X, test_labels)
predictions_svm_tfidf = svm_tfidf.predict(test_tfidf_X)


svm_count = SVC(kernel='linear', probability=True)
svm_count.fit(count_X, labels)

score_svm_count = svm_count.score(test_count_X, test_labels)
predictions_svm_count = svm_count.predict(test_count_X)


rf_tfidf = RandomForestClassifier(n_estimators=200, criterion='entropy')
rf_tfidf.fit(tfidf_X, labels)

score_rf_tfidf = rf_tfidf.score(test_tfidf_X, test_labels)
predictions_rf_tfidf = rf_tfidf.predict(test_tfidf_X)


rf_count = RandomForestClassifier(n_estimators=200, criterion='entropy')
rf_count.fit(count_X, labels)

score_rf_count = rf_count.score(test_count_X, test_labels)
predictions_rf_count = rf_count.predict(test_count_X)

pickle.dump(lgs_count, open('1lgs_count.pkl', 'wb'))
pickle.dump(lgs_tfidf, open('1lgs_tfidf.pkl', 'wb'))
pickle.dump(mnb_count, open('1mnb_count.pkl', 'wb'))
pickle.dump(mnb_tfidf, open('1mnb_tfidf.pkl', 'wb'))
pickle.dump(svm_count, open('1svm_count.pkl', 'wb'))
pickle.dump(svm_tfidf, open('1svm_tfidf.pkl', 'wb'))
pickle.dump(rf_count, open('1rf_count.pkl', 'wb'))
pickle.dump(rf_tfidf, open('1rf_tfidf.pkl', 'wb'))
