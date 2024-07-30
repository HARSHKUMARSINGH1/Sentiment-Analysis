#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report,  roc_curve, auc
import nltk
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure NLTK resources are downloaded
nltk.download('wordnet')
nltk.download('omw-1.4')


# In[8]:


DATASET_COLUMNS=['target','ids','date','flag','user','text']
DATASET_ENCODING = "ISO-8859-1"
df = pd.read_csv(r'C:\Users\harsh\OneDrive\Desktop\LANGUAGES\PYTHON\training.1600000.processed.noemoticon.csv', encoding=DATASET_ENCODING, names=DATASET_COLUMNS)


# In[9]:


df


# In[10]:


print("Unique value counts in the target column:")
print(df['target'].value_counts())


# In[18]:


import re
# Ensure you have downloaded the necessary NLTK data files
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

def clean_text(text):
    stopwordlist = [
        'a', 'about', 'above', 'after', 'again', 'ain', 'all', 'am', 'an',
        'and', 'any', 'are', 'as', 'at', 'be', 'because', 'been', 'before',
        'being', 'below', 'between', 'both', 'by', 'can', 'd', 'did', 'do',
        'does', 'doing', 'down', 'during', 'each', 'few', 'for', 'from',
        'further', 'had', 'has', 'have', 'having', 'he', 'her', 'here',
        'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in',
        'into', 'is', 'it', "it's", 'its', 'itself', 'just', 'll', 'm', 'ma',
        'me', 'more', 'most', 'my', 'myself', 'needn', 'no', 'nor', 'now',
        'o', 'of', 'on', 'once', 'only', 'or', 'other', 'our', 'ours', 'ourselves',
        'out', 'own', 're', 's', 'same', 'she', "shes", 'should', "shouldve", 'so', 'some', 'such',
        't', 'than', 'that', "thatll", 'the', 'their', 'theirs', 'them',
        'themselves', 'then', 'there', 'these', 'they', 'this', 'those',
        'through', 'to', 'too', 'under', 'until', 'up', 've', 'very', 'was',
        'we', 'were', 'what', 'when', 'where', 'which', 'while', 'who', 'whom',
        'why', 'will', 'with', 'won', 'y', 'you', "youd", "youll", "youre",
        "youve", 'your', 'yours', 'yourself', 'yourselves'
    ]

    # Function to get NLTK POS tag to WordNet POS tag
    def get_wordnet_pos(treebank_tag):
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN
    
    # Convert to lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'((www\.[^\s]+)|(https?://[^\s]+))', ' ', text)
    # Replace @mentions with 'USER'
    text = re.sub(r'@[\S]+', 'USER', text)
    # Remove hashtags but keep the text
    text = re.sub(r'#(\S+)', r'\1', text)
    # Remove digits
    text = re.sub(r'\d+', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Strip leading and trailing whitespace
    text = text.strip()
    # Remove stopwords
    text = " ".join([word for word in text.split() if word not in stopwordlist])
    # Tokenize text
    tokenizer = RegexpTokenizer(r'\w+|[^\w\s]')
    tokens = tokenizer.tokenize(text)

    # POS tagging
    pos_tags = nltk.pos_tag(tokens)

    # Lemmatize each token with the appropriate POS tag
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token, get_wordnet_pos(tag)) for token, tag in pos_tags]
    
    return " ".join(lemmatized_tokens)

# Example usage
text = "This is an example tweet! Check out https://example.com and contact me at @example_user."
cleaned_text = clean_text(text)
print(cleaned_text)


# In[19]:


# Replace target values
df['target'] = df['target'].replace(4, 1)

# Preprocess the text data
df['text'] = df['text'].apply(clean_text)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['target'], test_size=0.2, random_state=42)


# In[21]:


# Initialize the TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=500000, ngram_range=(1, 2))

# Fit and transform the training data
X_train_vect = vectorizer.fit_transform(X_train)

# Transform the testing data
X_test_vect = vectorizer.transform(X_test)


# In[23]:


# Step 5: Model building and evaluation function
def evaluate_model(model):
    model.fit(X_train_vect, y_train)
    y_pred = model.predict(X_test_vect)
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test_vect)[:, 1])
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc(fpr, tpr))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()


# In[24]:


# Step 6: Training and evaluating models
# Logistic Regression
lr_model = LogisticRegression(C = 2, max_iter = 1000, n_jobs=-1)
evaluate_model(lr_model)


# In[26]:


# Example test tweet
test_tweet = ["I love the new design of your website! It's too good and interactive!"]

test_tweet = [clean_text(test_tweet[0])]
# Transform the test tweet using the same vectorizer
vectorized_tweet = vectorizer.transform(test_tweet)

# Predict the sentiment
predicted_sentiment = lr_model.predict(vectorized_tweet)

# Output the result
print(f"Predicted Sentiment: {'Positive' if predicted_sentiment[0] == 1 else 'Negative'}")


# In[ ]:




