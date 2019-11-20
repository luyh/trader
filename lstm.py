import pandas_datareader.data as web
from stockstats import StockDataFrame as Sdf
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
import numpy
from alpha_vantage.timeseries import TimeSeries
highthresh = 0.9
lowthresh = 0.03
divtouse = 1
usesma = True

def get_data_from_csv(name,usesma=False):
 names = ['open', 'high', 'low', 'close', 'volume', 'momentum_rsi', 'adx',
       'adx_long', 'adx_pos', 'adx_neg', 'trend_ema_fast', 'trend_ema_slow']
 return pd.read_csv(name,names=names)



def get_data():
 data ='./data/processed/binance/btc_usdt_1h.csv'
 stock_df = pd.read_csv(data, index_col=[0])
 # data = web.get_data_yahoo(symbol)
 # stock_df = Sdf.retype(data)
 # stock_df.to_csv("example.csv",index=False)

 # data['trend_ema_slow'] = pd.Series(stock_df['trend_ema_slow'])
 # data['trend_ema_fast'] = pd.Series(stock_df['trend_ema_fast'])
 # data['adx'] = pd.Series(stock_df['adx'])
 # data['adx_long'] = pd.Series(stock_df['adx_long'])
 # data['adx_pos'] = pd.Series(stock_df['adx_pos'])
 # data['adx_neg'] = pd.Series(stock_df['adx_neg'])
 # data['momentum_rsi'] = pd.Series(stock_df['momentum_rsi'])

 return stock_df


def get_data_legnth(data):
 return len(data)


def divide_data(data):
 returnvec = []
 for i in range(0,get_data_legnth(data)):
  if i != 1:
   if i > 2:
    if get_data_legnth(data) % i == 0:
     returnvec.append(i)
 return returnvec
data = get_data()
stock = 'BTC/USDT'
#data = get_data_from_csv(stock+"_intraday.csv")
print(stock, len(data))
print(data)
data = data[2:1258]
datadivs = divide_data(data)
stock_df = Sdf.retype(data)
print(stock_df)
# data['trend_ema_slow'] = pd.Series(stock_df['trend_ema_slow'])
# data['trend_ema_fast'] = pd.Series(stock_df['trend_ema_fast'])
# data['adx'] = pd.Series(stock_df['adx'])
# data['adx_long'] = pd.Series(stock_df['adx_long'])
# data['adx_pos'] = pd.Series(stock_df['adx_pos'])
# data['adx_neg'] = pd.Series(stock_df['adx_neg'])
# data['momentum_rsi'] = pd.Series(stock_df['momentum_rsi'])
print(datadivs)

print(data.head())
print(len(data), get_data_legnth(data), datadivs[divtouse], datadivs[divtouse]*2 -2 )
npdata = numpy.random.random((int(get_data_legnth(data)/datadivs[divtouse]),datadivs[divtouse]*2 -2))


def format_data(data):
 begn = 0
 end = 0
 templist = []
 datareal = []
 resultreal = []
 pricelist = []
 d = datadivs[divtouse]
 pricelist = []
 i = 0
 for index,row in data.iterrows():
 #begn = row['close']

  if(i % datadivs[divtouse] == 0):
   if not(templist == []):
    datareal.append(templist)
   pricelist.append(row['close'])
   templist = []
   begn = float(row['close'])
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
   templist.append(row['trend_ema_slow'])
   templist.append(row['trend_ema_fast'])
   templist.append(row['adx'])
   templist.append(row['adx_long'])
   templist.append(row['adx_pos'])
   templist.append(row['adx_neg'])
   templist.append(row['momentum_rsi'])

  i = i +1
 return datareal,resultreal,pricelist



def fill(a,b,npdata,datafted):
 a = a-2
 b = b-1
 starta = a
 print(starta)
 try:
  while b >=0:

   while a >= 0:
    npdata[a][b] = datafted[a][b]
    a=a-1
   a = starta-1
   b=b-1
  return npdata
 except:
  print(b)
  return npdata




datafted,resultreal,pricelist = format_data(data)
npdata = fill(int(get_data_legnth(data)/datadivs[divtouse]),datadivs[divtouse]*2 -2,npdata,datafted)

def create_model(data,res,split, neurons=5, epochs=200):
 model = Sequential()
 model.add(Dense(128, input_dim=datadivs[divtouse]*2 -2, activation='relu'))
 model.add(Dropout(0.2))
 model.add(Dense(64, input_dim=datadivs[divtouse]*2 -2, activation='relu'))
 model.add(Dropout(0.2))
 model.add(Dense(34, input_dim=datadivs[divtouse]*2 -2, activation='relu'))
 model.add(Dropout(0.2))
 model.add(Dense(neurons, activation='relu'))
 # model.add(Dropout(0.2))
 model.add(Dense(1, activation='sigmoid'))
 model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
 model.fit(data[0:split],res[0:split], epochs=epochs, batch_size=10)
 #scores = model.evaluate(data[split:], res[split:])
 #print(scores)
 return model
md = create_model(npdata,resultreal,50,neurons=15,epochs=1000)

def predict(model,start,end,npdata):
 return model.predict(npdata[start:end])

predict(md,0, 50,npdata)
print('..... ')
print("real")
print(resultreal)
pred = predict(md,0, len(npdata),npdata)
pricelistft = []
for a in pricelist:
 pricelistft.append(float(a))
plt.plot(pricelistft)
def right_moves(resultreal,pred,pricelist):
 i = 0
 rm = 0
 wm = 0
 balance = 100
 consb = 100
 balancelist = []
 consblist = []
 for p in pred:
  if p[0] > 0.5:
   try:
    balance = balance * pricelistft[i+1]/pricelistft[i]
    plt.plot(i,pricelistft[i],'ro')
   except:
    pass
  else:
   try:
    balance = balance * pricelistft[i]/pricelistft[i+1]
    plt.plot(i,pricelistft[i],'g+')
   except:
    pass
  if resultreal[i] >0.5:
   if p[0] > 0.5:
    rm = rm +1

   else:
    wm = wm+1

   try:
    if(p[0] > highthresh):
     consb = consb * pricelistft[i+1]/pricelistft[i]


   except:
    pass
  else:
   if p[0] > 0.5:
    wm = wm+1
   else:
    rm = rm+1

   try:
    if(p[0] < lowthresh):
     consb = consb *pricelistft[i]/pricelistft[i+1]

   except:
    pass
  balancelist.append(balance)
  consblist.append(consb)
  i = i + 1

 return rm,wm,balance,consb,balancelist,consblist

rm,wm,balance,consb,balancelist,consblist = right_moves(resultreal,pred,pricelist)

print('current price(you might be on historical data)')

# ts = TimeSeries(key='DOI9SRTW04P19YQG', output_format='pandas')
#
# print(ts.get_quote_endpoint(stock))
# print('latest data trained on')
print(data.head())
print('right moves')
print(rm)
print('wrong moves')
print(wm)
print('balance')
print(balance)
print(consb)
plt.show()
plt.clf()
plt.plot(balancelist)
plt.show()
plt.clf()
plt.plot(consblist)
plt.show()
print(pred[-1])
print('using sma?')
print(usesma)
