# master-s-project
Master's project: A Deep Learning Application for credit scoring focusing on feature extraction and imbalance learning

With the rapid development of machine learning and artificial intelligence, an increasing number of applications have been developed to deal with the problem of credit scoring. In credit scoring, banks and loan lender are required to build a binary classification model to decide whether they will offer loans to their customers. In this project, in stead of popular tree-based models, a novel deep learning model is proposed solve this problem. The model first applies the attention mechanism, which helps the model to focus on important inputs. Then, the feature combination layer is designed to further exploit the latent relationship between features, thus increasing the feature dimension. Last, both synthetic minority oversampling technique (SMOTE) and focal loss function are adopted to deal with the problem of imbalance ratio, which is a prevalent problem in credit scoring datasets. To evaluate the performance of the model, four credit scoring datasets from UCI machine learning repository and two other open-sourced credit scoring dataset are studied. The performance of the proposed model is compared to other benchmark models and result shows that the proposed model outperforms many benchmark models based on various performance indicators.

Modules:
Python3.7
Tensorflow1.15
Keras
sklearn

How to use:
python run.py data train/test
(data=name of dataset)
