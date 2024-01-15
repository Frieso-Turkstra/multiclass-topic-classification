# multiclass-topic-classification

## Abstract

In this study, we tackle the task of multiclass topic classification for online reviews of six different products. We first implement a baseline system by experimenting with various LSTM networks. We experimented with both different architectures and different hyperparameters. Our best LSTM network employed a bi-directional architecture and achieved 89.8\% accuracy on the test set. Then, eight different PLMs were tested and the best one was chosen for further fine-tuning. The best model was BERT-cased which, after hyperparameter fine-tuning, achieved an accuracy of 95.2\% on the test set. This confirmed our hypothesis that PLMs would outperform our LSTM baseline system on the task of multiclass topic classification.

Check out the full article above in article.pdf. Note this was a group project and involved two other students.
