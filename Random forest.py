import numpy as np
import pandas
import datetime

df1=pandas.read_csv('new_data.csv')
dataset=df1.values
labels = np.array(dataset[0:, 4])
data = np.array(dataset[0:, :22])
np.delete(data,4,axis=1)

df2=pandas.read_csv('test_set.csv')
dataset_test=df2.values
labels_test = np.array(dataset_test[0:, 4])
data_test = np.array(dataset_test[0:, :22])
np.delete(data_test,4,axis=1)

starttime = datetime.datetime.now()

from sklearn.ensemble import RandomForestClassifier
cls = RandomForestClassifier(n_estimators=10)
cls = cls.fit(data, labels)

endtime = datetime.datetime.now()
print('The training time is', (endtime - starttime).microseconds/1000,'ms')

labels_pred = cls.predict(data_test)
print(labels_pred)
print(labels_test)

from sklearn.metrics import accuracy_score
s3=accuracy_score(labels_test,labels_pred)
print(s3)