from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier

app = Flask(__name__)
    
@app.route('/', methods=['GET', 'POST'])

def index():

    if request.method == 'POST':
        
        output = ""
        tags = ""
        
        #Read data
        data = pd.read_csv('data2.csv')
        data = data.drop(['Unnamed: 0'], axis = 1)

        def removeQuote(original_string):
            new_string = original_string.replace("[", "")
            new_string = new_string.replace("]", "")
            new_string = new_string.replace("'", "")
 
            return new_string

        data['Tags'] = data['Tags'].apply(lambda x: removeQuote(x))
        
        df= data.sample(n=10000, random_state = 42)

        tfidf = TfidfVectorizer(analyzer = 'word',
                        min_df=0.0,
                        max_df = 1.0,
                        strip_accents = None,
                        encoding = 'utf-8', 
                        preprocessor=None,
                        token_pattern=r"(?u)\S\S+",
                        max_features=1000)
        X = tfidf.fit_transform(df['sentence_bow_lem'])

             

        multilabel = MultiLabelBinarizer()
        y = df['Tags']
        y = multilabel.fit_transform(y.str.split(', '))
    
        
        X_train, X_test, y_train , y_test = train_test_split(X , y, test_size = 0.2, random_state= 0)
   
        svc = LinearSVC()
        clf = OneVsRestClassifier(svc)
        clf.fit(X_train, y_train)

        #Question from the user
        Question = request.form.get('fquestion')
        print(Question)
       
        def getTags(question):
            question = [question]
            question = tfidf.transform(question)
            tags = multilabel.inverse_transform(clf.predict(question))
            return tags
        
        try:
            out = getTags(Question)
            out1 = [item for t in out for item in t] 
            out1 = map(lambda e : "<" + e + ">", out1) 
            str1 = ","
            output = str1.join(out1)  
            if output == "":
                output = "Sorry, not tags to suggest!"    
            return render_template('index.html', tags=output, question=Question)
        except ValueError as e:
            print("error")
            return render_template('index.html', error=e)

    else:
        return render_template('index.html')
    
if __name__ == '__main__':
   app.run(debug = True)

