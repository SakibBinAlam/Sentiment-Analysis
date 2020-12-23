import re    # for regular expressions 
import nltk  # for text manipulation 
import string 
import warnings 
import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt  

pd.set_option("display.max_colwidth", 200) 
warnings.filterwarnings("ignore", category=DeprecationWarning) 

#%matplotlib inline

train  = pd.read_csv('train_E6oV3lV.csv') 
test = pd.read_csv('test_tweets_anuFYb8.csv')

# *** Data Inspection ***

#print(train[train['label'] == 0].head(10))
train[train['label'] == 1].head(10)

# dimension
train.shape
test.shape

# label distribution
train["label"].value_counts()

# distribution of length of the tweets, in terms of words
length_train = train['tweet'].str.len() 
length_test = test['tweet'].str.len() 
#plt.hist(length_train, bins=20, label="train_tweets") 
#plt.hist(length_test, bins=20, label="test_tweets") 
#plt.legend() 
#plt.show()

# *** Data Cleaning ***

# Before we begin cleaning, let’s first combine train and test datasets. 
# Combining the datasets will make it convenient for us to preprocess the data. 
# Later we will split it back into train and test data.
combi = train.append(test, ignore_index=True) 
combi.shape


#remove unwanted text patterns from the tweets
def remove_pattern(input_txt, pattern):
	r = re.findall(pattern, input_txt)
	for i in r:
		input_txt = re.sub(i, '', input_txt)
	return input_txt  

# removing @user. @[]* means any word starting with @
# create a new column tidy_tweet, it will contain the cleaned and processed tweets
combi['tidy_tweet'] = np.vectorize(remove_pattern)(combi['tweet'], "@[\w]*")
combi.head()

# Removing Punctuations, Numbers, and Special Characters
# Here we will replace everything except characters and hashtags with space
# regular expression “[^a-zA-Z#]” means anything except alphabets and ‘#’
combi['tidy_tweet'] = combi['tidy_tweet'].str.replace("[^a-zA-Z#]", " ") 
combi.head(10)

# removing short words
# I have decided to remove all the words having length 3 or less. For example, terms like “hmm”, “oh” 
# are of very little use. It is better to get rid of them
combi['tidy_tweet'] = combi['tidy_tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
combi.head()

# text normalization
tokenized_tweet = combi['tidy_tweet'].apply(lambda x: x.split()) # tokenizing
tokenized_tweet.head()

from nltk.stem.porter import * 
stemmer = PorterStemmer() 
tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x]) # stemming

# stitch these tokens back together
for i in range(len(tokenized_tweet)):
    tokenized_tweet[i] = ' '.join(tokenized_tweet[i])    
combi['tidy_tweet'] = tokenized_tweet

# *** Visualization from tweets ***

# common words in the tweets
all_words = ' '.join([text for text in combi['tidy_tweet']])
from wordcloud import WordCloud
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words) 
#plt.figure(figsize=(10, 7)) 
#plt.imshow(wordcloud, interpolation="bilinear") 
#plt.axis('off') 
#plt.show()

# words in non recist/sexist tweets
normal_words =' '.join([text for text in combi['tidy_tweet'][combi['label'] == 0]]) 
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(normal_words) 
#plt.figure(figsize=(10, 7)) 
#plt.imshow(wordcloud, interpolation="bilinear") 
#plt.axis('off') 
#plt.show()

# words in racist/sexist tweets
negative_words = ' '.join([text for text in combi['tidy_tweet'][combi['label'] == 1]]) 
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(negative_words) 
#plt.figure(figsize=(10, 7)) 
#plt.imshow(wordcloud, interpolation="bilinear") 
#plt.axis('off') 
#plt.show()

# hashtag/trend in twitter
# function to collect hashtags 
def hashtag_extract(x):    
	hashtags = []    
	# Loop over the words in the tweet    
	for i in x:        
		ht = re.findall(r"#(\w+)", i)        
		hashtags.append(ht)     
	return hashtags

# extracting hashtags from non racist/sexist tweets
HT_regular = hashtag_extract(combi['tidy_tweet'][combi['label'] == 0]) 
# extracting hashtags from racist/sexist tweets 
HT_negative = hashtag_extract(combi['tidy_tweet'][combi['label'] == 1]) 
# unnesting list 
HT_regular = sum(HT_regular,[]) 
HT_negative = sum(HT_negative,[])

# plot the top ‘n’ hashtags.
# non recist/sexist tweets
a = nltk.FreqDist(HT_regular) 
d = pd.DataFrame({'Hashtag': list(a.keys()),'Count': list(a.values())}) 
# selecting top 20 most frequent hashtags
d = d.nlargest(columns="Count", n = 20)
#plt.figure(figsize=(16,5)) 
ax = sns.barplot(data=d, x= "Hashtag", y = "Count") 
ax.set(ylabel = 'Count') 
#plt.show()

# recist/sexist tweets
b = nltk.FreqDist(HT_negative) 
e = pd.DataFrame({'Hashtag': list(b.keys()), 'Count': list(b.values())})
# selecting top 20 most frequent hashtags
e = e.nlargest(columns="Count", n = 20)   
#plt.figure(figsize=(16,5)) 
ax = sns.barplot(data=e, x= "Hashtag", y = "Count")
#plt.show()

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer 
import gensim

# ***Bag-of-Words feature***

bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
bow = bow_vectorizer.fit_transform(combi['tidy_tweet']) 
bow.shape

# ***TF-IDF feature***
tfidf_vectorizer = TfidfVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(combi['tidy_tweet']) 
tfidf.shape

# ***Word2Vec feature***

# There are some freely pre trained vectors
# Here we will train our own word vectors since size of the pre-trained word vectors is generally huge
tokenized_tweet = combi['tidy_tweet'].apply(lambda x: x.split()) # tokenizing
model_w2v = gensim.models.Word2Vec(
            tokenized_tweet,
            size=200, # desired no. of features/independent variables
            window=5, # context window size
            min_count=2,
            sg = 1, # 1 for skip-gram model
            hs = 0,
            negative = 10, # for negative sampling
            workers= 2, # no.of cores
            seed = 34)
model_w2v.train(tokenized_tweet, total_examples= len(combi['tidy_tweet']), epochs=20)

# We will specify a word and the model will pull out the most similar words from the corpus
#print(model_w2v.wv.most_similar(positive="dinner"))
#model_w2v.wv.most_similar(positive="trump")

# vector representation of any word from our corpus
model_w2v['food']
len(model_w2v['food'])

# preparing vectors for tweets
# we can simply take mean of all the word vectors present in the tweet. The length of the resultant vector will be the same,
# i.e. 200. We will repeat the same process for all the tweets in our data and obtain their vectors. 
# Now we have 200 word2vec features for our data.

# function to create a vector for each tweet by taking the average of the vectors of the words present in the tweet
def word_vector(tokens, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0
    for word in tokens:
        try:
            vec += model_w2v[word].reshape((1, size))
            count += 1.
        except KeyError: #handling the case where the token is not in vocabulary
        	continue

    if count != 0:
        vec /= count
    return vec

# preparinh word2vec feature set
wordvec_arrays = np.zeros((len(tokenized_tweet), 200)) 
for i in range(len(tokenized_tweet)):
    wordvec_arrays[i,:] = word_vector(tokenized_tweet[i], 200)
    wordvec_df = pd.DataFrame(wordvec_arrays)

wordvec_df.shape


# Now we have 200 new features, whereas in Bag of Words and TF-IDF we had 1000 features.
""""
# ***Doc2Vec Embedding***

# generate vectors for sentence/paragraphs/documents
# The major difference between the two is that doc2vec provides an additional context which is unique for every document in the corpus
from tqdm import tqdm #tqdm.pandas(desc="progress-bar") 
from gensim.models.doc2vec import LabeledSentence

# To implement doc2vec, we have to labelise or tag each tokenised tweet with unique IDs. We can do so by 
# using Gensim’s LabeledSentence() function
def add_label(twt):
    output = []
    for i, s in zip(twt.index, twt):
        output.append(LabeledSentence(s, ["tweet_" + str(i)]))
    return output
labeled_tweets = add_label(tokenized_tweet) # label all the tweets

labeled_tweets[:6] # print it

# train a doc2vec model
model_d2v = gensim.models.Doc2Vec(dm=1, # dm = 1 for ‘distributed memory’ model
	dm_mean=1, # dm = 1 for using mean of the context word vectors
	size=200, # no. of desired features  
	window=5, # width of the context window
	negative=7, # if > 0 then negative sampling will be used
	min_count=5, # Ignores all words with total frequency lower than 2
	workers=3, # no. of cores
	alpha=0.1, # learning rate
	seed = 23) 
model_d2v.build_vocab([i for i in tqdm(labeled_tweets)])
model_d2v.train(labeled_tweets, total_examples= len(combi['tidy_tweet']), epochs=15)

# Preparing doc2vec Feature Set
docvec_arrays = np.zeros((len(tokenized_tweet), 200)) 
for i in range(len(combi)):
    docvec_arrays[i,:] = model_d2v.docvecs[i].reshape((1,200))    

docvec_df = pd.DataFrame(docvec_arrays) 
docvec_df.shape """

# *** Modeling ***

# ***Logistic Regression***

from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import f1_score

# --- Bag-of-Words feature ---. First we will try with Bag-of-Words feature

# Extracting train and test Bow features
train_bow = bow[:31962,:] 
test_bow = bow[31962:,:]
# splitting data into training and validation set
xtrain_bow, xvalid_bow, ytrain, yvalid = train_test_split(train_bow, train['label'],random_state=42,test_size=0.3)

lreg = LogisticRegression()
# training the model
lreg.fit(xtrain_bow, ytrain)
prediction = lreg.predict_proba(xvalid_bow) # predicting on the validation set
prediction_int = prediction[:,1] >= 0.3 # if prediction is greater than or equal to 0.3 than 1 else 0
prediction_int = prediction_int.astype(np.int) 
f1_score(yvalid, prediction_int) # calculating f1 score for the validation set. print it

# let’s make predictions for the test dataset and create a submission file
test_pred = lreg.predict_proba(test_bow)
test_pred_int = test_pred[:,1] >= 0.3
test_pred_int = test_pred_int.astype(np.int)
test['label'] = test_pred_int
submission = test[['id','label']]
submission.to_csv('sub_lreg_bow.csv', index=False) # writing data to a CSV file


# --- TF-IDF features ---

train_tfidf = tfidf[:31962,:]
test_tfidf = tfidf[31962:,:]
xtrain_tfidf = train_tfidf[ytrain.index]
xvalid_tfidf = train_tfidf[yvalid.index]

lreg.fit(xtrain_tfidf, ytrain) 
prediction = lreg.predict_proba(xvalid_tfidf)
prediction_int = prediction[:,1] >= 0.3
prediction_int = prediction_int.astype(np.int)
f1_score(yvalid, prediction_int) # f1 = 0.544 . on test set f1 = 0.564

# --- Word2Vec features ---

train_w2v = wordvec_df.iloc[:31962,:] 
test_w2v = wordvec_df.iloc[31962:,:] 
xtrain_w2v = train_w2v.iloc[ytrain.index,:] 
xvalid_w2v = train_w2v.iloc[yvalid.index,:]

lreg.fit(xtrain_w2v, ytrain)
prediction = lreg.predict_proba(xvalid_w2v)
prediction_int = prediction[:,1] >= 0.3
prediction_int = prediction_int.astype(np.int)
print(f1_score(yvalid, prediction_int)) #f1 score = 0.622. on test set f1 = 0.661

# --- Doc2Vec Features ---

train_d2v = docvec_df.iloc[:31962,:] 
test_d2v = docvec_df.iloc[31962:,:] 
xtrain_d2v = train_d2v.iloc[ytrain.index,:] 
xvalid_d2v = train_d2v.iloc[yvalid.index,:]

lreg.fit(xtrain_d2v, ytrain)
prediction = lreg.predict_proba(xvalid_d2v) 
prediction_int = prediction[:,1] >= 0.3
prediction_int = prediction_int.astype(np.int) 
f1_score(yvalid, prediction_int) # f1 = 0.361 . on test set f1 = 0.381

# *** SVM ***

from sklearn import svm
# --- Bag-of-Words Features ---

svc = svm.SVC(kernel='linear', C=1, probability=True).fit(xtrain_bow, ytrain) 
prediction = svc.predict_proba(xvalid_bow) 
prediction_int = prediction[:,1] >= 0.3 
prediction_int = prediction_int.astype(np.int) 
f1_score(yvalid, prediction_int) # f1 = 0.508

# prediction for the test data set
test_pred = svc.predict_proba(test_bow) 
test_pred_int = test_pred[:,1] >= 0.3 
test_pred_int = test_pred_int.astype(np.int) 
test['label'] = test_pred_int 
submission = test[['id','label']] 
submission.to_csv('sub_svm_bow.csv', index=False) # f1 = 0.554

# --- TF-IDF Features ---

svc = svm.SVC(kernel='linear', C=1, probability=True).fit(xtrain_tfidf, ytrain)
prediction = svc.predict_proba(xvalid_tfidf) 
prediction_int = prediction[:,1] >= 0.3 
prediction_int = prediction_int.astype(np.int) 
f1_score(yvalid, prediction_int) # f1 = 0.51

# --- Word2Vec Features ---

svc = svm.SVC(kernel='linear', C=1, probability=True).fit(xtrain_w2v, ytrain) 
prediction = svc.predict_proba(xvalid_w2v) 
prediction_int = prediction[:,1] >= 0.3 
prediction_int = prediction_int.astype(np.int) 
f1_score(yvalid, prediction_int) # f1 = 0.614

# --- Doc2Vec Features ---

svc = svm.SVC(kernel='linear', C=1, probability=True).fit(xtrain_d2v, ytrain) 
prediction = svc.predict_proba(xvalid_d2v) 
prediction_int = prediction[:,1] >= 0.3 
prediction_int = prediction_int.astype(np.int) 
f1_score(yvalid, prediction_int) # f1 = 0.203

# *** RandomForest ***

from sklearn.ensemble import RandomForestClassifier

# --- Bag-of-Words Features ---

rf = RandomForestClassifier(n_estimators=400, random_state=11).fit(xtrain_bow, ytrain) 
prediction = rf.predict(xvalid_bow)
f1_score(yvalid, prediction) # f1 = 0.553

# predictions for the test dataset and create another submission file
test_pred = rf.predict(test_bow)
test['label'] = test_pred
submission = test[['id','label']]
submission.to_csv('sub_rf_bow.csv', index=False) # f1 = 0.589

# --- TF-IDF Features ---

rf = RandomForestClassifier(n_estimators=400, random_state=11).fit(xtrain_tfidf, ytrain) 
prediction = rf.predict(xvalid_tfidf) 
f1_score(yvalid, prediction) # f1 = 0.562

# --- Word2Vec Features ---

rf = RandomForestClassifier(n_estimators=400, random_state=11).fit(xtrain_w2v, ytrain) 
prediction = rf.predict(xvalid_w2v) 
f1_score(yvalid, prediction) # f1 = 0.507

# --- Doc2Vec Features ---

rf = RandomForestClassifier(n_estimators=400, random_state=11).fit(xtrain_d2v, ytrain) 
prediction = rf.predict(xvalid_d2v) 
f1_score(yvalid, prediction) # f1 = 0.056

# *** XGBOOST ***

from xgboost import XGBClassifier

# --- Bag-of-Words Features ---

xgb_model = XGBClassifier(max_depth=6, n_estimators=1000).fit(xtrain_bow, ytrain)
prediction = xgb_model.predict(xvalid_bow) 
f1_score(yvalid, prediction) # f1 = 0.513

# predictions for the test dataset and create another submission file
est_pred = xgb_model.predict(test_bow) 
test['label'] = test_pred
submission = test[['id','label']] 
submission.to_csv('sub_xgb_bow.csv', index=False) # f1 = 0.554

# --- TF-IDF Features ---

xgb = XGBClassifier(max_depth=6, n_estimators=1000).fit(xtrain_tfidf, ytrain) 
prediction = xgb.predict(xvalid_tfidf) 
f1_score(yvalid, prediction) #f1 = 0.554

# --- Word2Vec Features ---

xgb = XGBClassifier(max_depth=6, n_estimators=1000, nthread= 3).fit(xtrain_w2v, ytrain) 
prediction = xgb.predict(xvalid_w2v) 
f1_score(yvalid, prediction) #f1 = 0.652

# --- Doc2Vec Features ---

xgb = XGBClassifier(max_depth=6, n_estimators=1000, nthread= 3).fit(xtrain_d2v, ytrain) 
prediction = xgb.predict(xvalid_d2v) 
f1_score(yvalid, prediction) #f1 = 0.345

# XGBoost with Word2Vec model has given us the best performance so far. Try to tune it 
# further to extract as much from it as we can.


