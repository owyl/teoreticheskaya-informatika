import csv
import codecs
with codecs.open('news_train.txt','r', 'utf-8') as fin_train:
     train_read = csv.reader(fin_train, delimiter='\t')
     train_categories = []
     train_headers = []
     train_text = []
     for row in train_read:
         category = row[0]
         header = row[1]
         text = row[2]
         train_categories.append(category)
         train_headers.append(header)
         train_text.append(text)
with codecs.open('news_test.txt', 'r', 'utf-8') as fin_test:
     test_read = csv.reader(fin_test, delimiter='\t')
     test_headers = []
     test_text = []
     for row in test_read:
         header = row[0]
         text = row[1]
         test_headers.append(header)
         test_text.append(text)

train_article = []
test_article = []
i=0
for element in train_text:
     train_text[i]=train_headers[i] + ' ' + train_text[i]
     i=i+1
i=0
for element in test_text:
     test_text[i]=test_headers[i] + ' ' + test_text[i]
     i=i+1
	 
from sklearn.cross_validation import train_test_split

x_train, x_test, target_train, target_test = train_test_split(train_text, train_categories)

from sklearn import linear_model
logreg = linear_model.LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
classifier = Pipeline([
	('vectorizer', CountVectorizer(1, 2)),
	('tfidf', TfidfTransformer()),
	('clf', OneVsRestClassifier(logreg()))])
classifier.fit(x_train, target_train)

classifier.score(x_test, target_test)
#0.88339999999999996

classifier.fit(train_text, train_categories)
result = classifier.predict(test_text)
with codecs.open('news_output_polyanskiy.txt', 'w', 'utf-8') as fout:
	for element in result:
		fout.write(element+'\n')