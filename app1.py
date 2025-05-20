from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from flask import Flask, request, render_template
from sklearn.model_selection import train_test_split
import pickle
import pandas as pd
import re


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
cVec = CountVectorizer(tokenizer=tokenizer)
tVec = TfidfVectorizer(tokenizer=tokenizer)
cVec.fit_transform(train_df['URLs'])
tVec.fit_transform(train_df['URLs'])

feature_names = tVec.get_feature_names_out()


def get_words(text):
    tfidf_matrix = tVec.transform([text]).todense()
    feature_index = tfidf_matrix[0, :].nonzero()[1]
    tfidf_scores = zip([feature_names[i] for i in feature_index], [tfidf_matrix[0, x] for x in feature_index])
    return dict(tfidf_scores)


lgs_count = pickle.load(open('1lgs_count.pkl', 'rb'))
lgs_tfidf = pickle.load(open('1lgs_tfidf.pkl', 'rb'))
mnb_count = pickle.load(open('1mnb_count.pkl', 'rb'))
mnb_tfidf = pickle.load(open('1mnb_tfidf.pkl', 'rb'))
svm_count = pickle.load(open('1svm_count.pkl', 'rb'))
svm_tfidf = pickle.load(open('1svm_tfidf.pkl', 'rb'))
rf_count = pickle.load(open('1rf_count.pkl', 'rb'))
rf_tfidf = pickle.load(open('1rf_tfidf.pkl', 'rb'))


def get_output(url, vec, ml):
    pred=0
    prob=0
    if vec==1 and ml==1:
        example_Xc = cVec.transform([url])
        example_X = example_Xc
        lgs_c_pred = lgs_count.predict(example_X)
        lgs_c_pro = max(lgs_count.predict_proba(example_X)[0])
        pred = lgs_c_pred
        prob = lgs_c_pro

    elif vec==1 and ml==2:
        example_Xc = cVec.transform([url])
        example_X = example_Xc
        mnb_c_pred = mnb_count.predict(example_X)
        mnb_c_pro = max(mnb_count.predict_proba(example_X)[0])
        pred = mnb_c_pred
        prob = mnb_c_pro

    elif vec==1 and ml==3:
        example_Xc = cVec.transform([url])
        example_X = example_Xc
        rf_c_pred = rf_count.predict(example_X)
        rf_c_pro = max(rf_count.predict_proba(example_X)[0])
        pred = rf_c_pred
        prob = rf_c_pro

    elif vec==1 and ml==4:
        example_Xc = cVec.transform([url])
        example_X = example_Xc
        svm_c_pred = svm_count.predict(example_X)
        svm_c_pro = max(svm_count.predict_proba(example_X)[0])
        pred = svm_c_pred
        prob = svm_c_pro

    elif vec==2 and ml==1:
        example_Xt = tVec.transform([url])
        example_X = example_Xt
        lgs_t_pred = lgs_tfidf.predict(example_X)
        lgs_t_pro = max(lgs_tfidf.predict_proba(example_X)[0])
        pred = lgs_t_pred
        prob = lgs_t_pro

    elif vec==2 and ml==2:
        example_Xt = tVec.transform([url])
        example_X = example_Xt
        rf_t_pred = rf_tfidf.predict(example_X)
        rf_t_pro = max(rf_tfidf.predict_proba(example_X)[0])
        pred = rf_t_pred
        prob = rf_t_pro

    elif vec==2 and ml==3:
        example_Xt = tVec.transform([url])
        example_X = example_Xt
        rf_t_pred = rf_tfidf.predict(example_X)
        rf_t_pro = max(rf_tfidf.predict_proba(example_X)[0])
        pred = rf_t_pred
        prob = rf_t_pro

    elif vec==2 and ml==4:
        example_Xt = tVec.transform([url])
        example_X = example_Xt
        svm_t_pred = svm_tfidf.predict(example_X)
        svm_t_pro = max(svm_tfidf.predict_proba(example_X)[0])
        pred = svm_t_pred
        prob = svm_t_pro

    return (pred,prob, vec, ml)



app = Flask(__name__)


@app.route("/")
def Home():
    return render_template("Home2.html")


@app.route("/info")
def info():
    return render_template("Info.html")


@app.route("/about")
def about():
    return render_template("About.html")


@app.route("/contactus")
def contact():
    return render_template("Contact1.html")


@app.route("/classify/")
def classify():
    return render_template("Prediction.html")


@app.route("/predict/", methods=["POST"])
def predict():

    res = [x for x in request.form.values()]
    url = res[-1]
    vec = int(res[0])
    ml = int(res[1])

    pred, prob, vec, ml = get_output(url, vec, ml)

    if vec==1 and ml==1:
        v = 'Count Vectorizer'
        m = 'Logistic Regresion model'

    elif vec==1 and ml==2:
        v = 'Count Vectorizer'
        m = 'Multinomial Naive Bayes'

    elif vec==1 and ml==3:
        v = 'Count Vectorizer'
        m = 'Random Forest Classifier'

    elif vec==1 and ml==4:
        v = 'Count Vectorizer'
        m = 'Support Vector Machine'

    elif vec==2 and ml==1:
        v = 'TFIDF Vectorizer'
        m = 'Logistic Regresion model'

    elif vec==2 and ml==2:
        v = 'TFIDF Vectorizer'
        m = 'Multinomial Naive Bayes'

    elif vec==2 and ml==3:
        v = 'TFIDF Vectorizer'
        m = 'Random Forest Classifier'

    elif vec==2 and ml==4:
        v = 'TFIDF Vectorizer'
        m = 'Support Vector Machine'

    df = pd.DataFrame()
    df1 = {'URL': url, 'Vectorizer': vec, 'ML Model': ml, 'Prediction': pred, 'Probability': prob}
    df = pd.DataFrame([df1])

    #df1 = {'URL':url, 'Vectorizer':vec, 'ML Model':ml, 'Prediction':pred, 'Probability':prob}
    #df = df.append(df1, ignore_index=True)
    dd = get_words(url)
    dp = pd.DataFrame(dd, index=[0]).T
    return render_template("Prediction.html", prediction_text="By using {0:} and {1:} we found that the given website {2:} is '{3:}' and"
        " probability of being '{3:}' is {4:0.4f}".format(v, m, url, pred[0], prob))
        # tables=[df.to_html(classes='data', header=False, justify='center')], titles=df.columns.values)


if __name__ == "__main__":
    app.run(debug=True)
