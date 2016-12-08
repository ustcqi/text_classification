This project mainly focus on the topic of text classification, including english and chinese text.

Several open project have been used, such as Scikit-Learn, Gensim, Numpy, Pandas, Matplotlib.

Processing Steps of Text Classification:
Step 1: Data Analysis
1, distribution of all categories
2, the average lengith of each catogory of the text

Step 2: Text Preprocessing
Tokenization:
1, remove the punctuation
2, hyphenation
3, remove stop words
4, remove high-frequency and low-frequency words

Normalization:
1, case-folding
2, stemming

Step 3: Feature Selection
1, L1 Reg and IG ,removing stop words
2, Removing the words appearance in all the documents and with high frequency

Step 4: Represention of the text
1, tfidf
2, Bag-of-words
3, One-Hot
4, word2vec

Step 5: Regularization
L1 or L2

Step 6: Search for Hyperparameters
GridSearch

Step 7: Cross-Validation
10 fold cross-validation

Step 8: Machine Learning Pipeline

