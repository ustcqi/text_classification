##Description:
This project mainly focus on the topic of text classification, including english and chinese text.<br />
Several open project have been used, such as Scikit-Learn, Gensim, Numpy, Pandas, Matplotlib. <br />

##Evironment:
Python 2.7 <br />
Scikit-learn 0.18 <br />

##Usage:
python text_classification 

##Result:
>text_classification/src/result <br />
The following figure shows that if we using the BoW represention with our project dataset, the Logistic Regression has the best performance.
>![image](https://github.com/ustcqi/text_classification/blob/master/src/result/count.png) <br />

<br />
If we using the tfidf represention the SVM has the best performance.
>![image](https://github.com/ustcqi/text_classification/blob/master/src/result/tfidf.png) <br />
When putting BoW and tfidf togethor, the SVM-tfidf model has a 0.9625 accuracy, however LR-count's accuracy is 0.9607 a bit worse than SVM-tfidf. <br />
>text_cls.log <br />
>comparison.csv <br />

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
>1, tfidf <br />
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

###Extension:
####Ensembles:
1, Random Forest(Feature selection) + LR/SVM <br />
2, GBDT(stacking) + LR/SVM <br />
3, SVM + CNN  <br />
4, SVM + CNN + RNN <br />
