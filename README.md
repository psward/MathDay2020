# MathDay2020

Above are my two, three minute talks for Math Day 2020.

```python
import pandas as pd

df = pd.read_csv('processed_article_jsons.csv')
df.dropna(inplace=True)
```


```python
from sklearn.feature_extraction.text import CountVectorizer
from operator import itemgetter
from nltk.tokenize import word_tokenize 
import re
import numpy as np

def tokenizer(s):
    res = []
    words = word_tokenize(s)
    for w in words:
        w = re.sub("[^A-Za-z]", "", w) 
        if len(w) < 2:
            continue
        else:
            res.append(w)
    return res

cv = CountVectorizer(stop_words='english', tokenizer=tokenizer)
td = cv.fit_transform(df['abstract'].values)
```


```python
frequencies1 = dict(zip(cv.get_feature_names(), np.ravel(td.sum(axis=0))))
num_of_keys1 = len(frequencies1.keys())

res1 = dict(sorted(frequencies1.items(), key = itemgetter(1), reverse = True)[:138])
stop_words_by_freq1 = [k for k,v in res1.items() if v >= 5000]
```


```python
frequencies2 = dict(zip(cv.get_feature_names(), np.ravel(td.sum(axis=0))))
num_of_keys2 = len(frequencies2.keys())

res2 = dict(sorted(frequencies2.items(), key = itemgetter(1), reverse = False))
stop_words_by_freq2 = [k for k,v in res2.items() if v <= 20]
```


```python
%%time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
from nltk.tokenize import word_tokenize 
from nltk.stem import WordNetLemmatizer
import re

wnl = WordNetLemmatizer()

def isplural(word):
    lemma = wnl.lemmatize(word, 'n')
    plural = True if word is not lemma else False
    return plural, lemma


def tokenizer(s):
    res = []
    words = word_tokenize(s)
    for w in words:
        w = re.sub("[^A-Za-z]", "", w) 
        if len(w) < 2:
            continue
        if w in stop_words:
            continue
        else:
            check, lemma = isplural(w)
            res.append(lemma)
    return res

hf_stop_words = ['results', 'protein', 'study',
                    'cells', 'infection', 'viruses', 'viral', 'et', 'al', 'acute', 'severe']

stop_words = text.ENGLISH_STOP_WORDS.union(stop_words_by_freq2, hf_stop_words)


#jvocab = {"incubation":0,"shedding":1,"transmission":2,"adhesion":3,"substrate":4,"surface":5,"response":6,"seasonality":7,"risk":8,"contagious":9}

vectorizer = TfidfVectorizer(ngram_range=(2,2), max_features=10, 
                             sublinear_tf=False, tokenizer=tokenizer, stop_words=stop_words)
X = vectorizer.fit_transform(df['abstract'].values)
```

    CPU times: user 41.5 s, sys: 256 ms, total: 41.8 s
    Wall time: 41.8 s



```python
from matplotlib import pyplot as plt
import seaborn as sns

sns.set(rc={'figure.figsize':(12,12)})

plt.scatter(X.toarray()[:,0], X.toarray()[:,1])
plt.gca().set_aspect('equal', 'datalim')
```


![png](output_5_0.png)



```python
from sklearn.decomposition import PCA

pca = PCA(n_components=0.95)
X_reduced= pca.fit_transform(X.toarray())
X_reduced.shape
```




    (23645, 9)




```python
from matplotlib import pyplot as plt

plt.scatter(X_reduced[:, 0], X_reduced[:, 1])
plt.gca().set_aspect('equal', 'datalim')
```


![png](output_7_0.png)



```python
import umap
import warnings
warnings.filterwarnings('ignore')

reducer = umap.UMAP(n_neighbors=5, min_dist = 0.1, metric='cosine')
embedding = reducer.fit_transform(X_reduced)
embedding.shape
```




    (23645, 2)




```python
from matplotlib import pyplot as plt

plt.scatter(embedding[:, 0], embedding[:, 1])
plt.gca().set_aspect('equal', 'datalim')
```


![png](output_9_0.png)



```python
from sklearn.mixture import GaussianMixture

k = 8
gmm = GaussianMixture(n_components=k)
y_pred = gmm.fit_predict(embedding)
y = y_pred
```


```python
from matplotlib import pyplot as plt
import seaborn as sns

sns.set(rc={'figure.figsize':(12,12)})
palette = sns.hls_palette(k, l=.4, s=.9)

sns.scatterplot(embedding[:,0], embedding[:,1], hue=y_pred, legend='full', palette=palette)
plt.show()
```


![png](output_11_0.png)



```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X.toarray(),y_pred, test_size=0.2, random_state=42)

print("X_train size:", len(X_train))
print("X_test size:", len(X_test), "\n")
```

    X_train size: 18916
    X_test size: 4729 
    



```python
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_score, recall_score

forest_clf = RandomForestClassifier(n_jobs=-1)

forest_scores = cross_val_score(forest_clf, X_train, y_train, cv=3, n_jobs=-1)
forest_train_pred = cross_val_predict(forest_clf, X_train, y_train, cv=3, n_jobs=-1)

print("Precision: ", '{:,.3f}'.format(float(precision_score(y_train, forest_train_pred, average='macro')) * 100), "%")
print("   Recall: ", '{:,.3f}'.format(float(recall_score(y_train, forest_train_pred, average='macro')) * 100), "%")
print(" Accuracy: ", '{:,.3f}'.format(float(forest_scores.mean()) * 100), "%")
```

    Precision:  97.818 %
       Recall:  98.616 %
     Accuracy:  99.154 %



```python

```
