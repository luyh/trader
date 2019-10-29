import pandas_datareader.data as web
data = web.get_data_yahoo('spy')
from stockstats import StockDataFrame as Sdf
import pandas as pd
stock_df = Sdf.retype(data)
stock_df.to_csv("example.csv",index=False)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import numpy
data['macd'] = pd.Series(stock_df['macd'])
print(data)

datatokrs = data.values
dt = data

neww = data[['close','macd']].copy()
print(neww)
divs = 17
begn = 0
end = 0
i = 0
resultreal = []
datareal = []
templist = []

something = numpy.random.random((74,32))

for index,row in data.iterrows():
 #begn = row['close']

 if(i % divs == 0):
  if not(templist == []):
   datareal.append(templist)
  templist = []
  begn = row['close']
  if(begn > end):
   print('up')
   resultreal.append(1)
   end = begn
  else:
   print('down')
   resultreal.append(0)
   end = begn
 else:
  templist.append(row['close'])
  templist.append(row['macd'])

 i = i +1
#print(resultreal)

print(len(datareal[1]))
i = 31
j = 72
while(j >= 0):
 while(i >= 0):
  something[j][i] = datareal[j][i]
  #print(something[j][i])
  #print(datareal[j][i])
  i = i - 1
 i = 31
 j = j - 1

#print(something[0][0])
#print(datareal[0][0])
npresre = numpy.array(resultreal)
#print(npresre)
model = Sequential()
model.add(Dense(34, input_dim=32, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(something[0:50],resultreal[0:50], epochs=150, batch_size=10)
scores = model.evaluate(something[50:74], resultreal[50:74])
print(scores)
modelres = model.predict(something)
i = 0
tot = 0
right = 0
print(modelres[50:74])
for res in modelres:
 print(res[0])
 print(resultreal[i])
 if(res[0] - resultreal[i] > 0.5 or res[0] - resultreal[i] < -0.5):
  print('wrong')
 else:
  print("right")
  right = right + 1
 i = i +1
 tot = tot+1
print("acc : ")
print(right/tot)
