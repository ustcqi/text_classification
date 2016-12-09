##Description:
This project mainly focus on the topic of text classification, including the english and chinese text.<br />
Several open projects have been used, such as Scikit-Learn, NLTK, Gensim, Numpy, Pandas, Matplotlib. <br />

##Evironment:
Python 2.7 <br />
Scikit-learn 0.18 <br />
NLTK 3.2.1<br />

##Usage:
python text_classification.py 

##Result:
text_classification/src/result <br />
The following figure shows that if we using the BOW represention with our project dataset, the Logistic Regression has the best performance.
>![image](https://github.com/ustcqi/text_classification/blob/master/src/result/count.png) <br />

<br />
If we using the TFIDF represention, the SVM has the best performance.
![image](https://github.com/ustcqi/text_classification/blob/master/src/result/tfidf.png) <br />
When putting BOW and TFIDF togethor, the SVM-TFIDF model has a 0.9625 accuracy a bit better than LR-BOW which accuracy is 0.9607. <br />
All results in the following file.
>src/result/comparison.csv <br />

##Processing Steps of Text Classification:
####Step 1: Data Analysis
>1, distribution of all categories <br/>
>2, the average lengith of each catogory of the text<br />

####Step 2: Text Preprocessing
Tokenization:
>1, remove the punctuation <br />
>2, hyphenation <br/>
>3, remove stop words <br/>
>4, remove high-frequency and low-frequency words <br />

Normalization:
>1, case-folding <br />
>2, stemming <br />

####Step 3: Feature Selection
>1, L1 Reg and IG ,removing stop words <br />
>2, Removing the words appearance in all the documents and with high frequency <br />

####Step 4: Represention of the text
>1, TFIDF <br />
>2, Bag-of-words <br />
>3, One-Hot <br />
>4, word2vec <br />

####Step 5: Regularization
>L1 or L2 <br />

####Step 6: Search for Hyperparameters
>GridSearch <br />

####Step 7: Cross-Validation
>10 fold cross-validation <br />

####Step 8: Machine Learning Pipeline

###Extension and Further plan:
####Ensembles:(not used in this project)
1, Random Forest(Feature selection) + LR/SVM <br />
2, GBDT(stacking) + LR/SVM <br />
3, SVM + CNN  <br />
4, SVM + CNN + RNN <br />

####Further plan 
1, Bayes Optimization for seaching the Hyperparameters instead of GridSearch.

