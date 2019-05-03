import os
import numpy as np
import pandas as pd
from urllib import request
import zipfile
import datetime
import statsmodels.api as st
import matplotlib
import matplotlib as plt
import seaborn as sns



csv_file = open('info.csv', 'w')
csv_file.write('Time,CIK,Company,8-K filing date,path\n')

for y in range(1995,2017):
    for q in range(1,5):

        print(y, ' ', q)
        
        path = str(y) + 'Q' + str(q)
        url = 'https://www.sec.gov/Archives/edgar/full-index/' + str(y) + '/QTR' + str(q) + '/master.zip'
        request.urlretrieve(url, 'master.zip')

        z = zipfile.ZipFile('master.zip')
        z.extract('master.idx')
        z.close()

        file = open('master.idx','r')
        for i in range(11):
            s = file.readline()
            s = s[:-1]
            if (i == 9  and  s != 'CIK|Company Name|Form Type|Date Filed|Filename'):
                print('error')
            if (i == 10  and  s != '--------------------------------------------------------------------------------'):
                print('error2')

        companies = []
        for s in file:
            if (len(s) == 0):
                print('error4')
            if (s[-1] == '\n'):
                s = s[:-1]
            l = s.split('|')
            l[1] = l[1].replace(',','')
            if (l[2] == '8-K'):
                companies.append([l[0], l[1], l[3], l[4]])

        file.close()
        os.remove('master.zip')
        os.remove('master.idx')

        sample_n = 100
        if (len(companies) < sample_n):
            print('error3')
        ind = np.random.choice(len(companies), sample_n, False)
        companies = np.array(companies)
        companies = companies[ind,:]

        
        os.mkdir(path)
        for i in range(sample_n):
            url = 'https://www.sec.gov/Archives/' + companies[i][3]
            s = companies[i][3]
            s = s[-24:]
            request.urlretrieve(url, path + '/' + s)

        for i in range(sample_n):
            s = companies[i][3]
            s = s[-24:]
            if (not os.path.isfile(path + '/' + s)):
                print('error6 ' + s)
        

        for l in companies:
            csv_file.write('%s,%s,%s,%s,%s\n' % (path, l[0], l[1], l[2], l[3]))

csv_file.close()

#################################################################################
#################################################################################
'''                                                                             
Please refer to DC.SAS
The SAS code needs to be ran to generate CUSIP and DSF in order to proceed

'''
################################################################################
################################################################################

dsf = pd.read_csv('dsf.csv', dtype={'CUSIP':str, 'DATE':np.int64, 'RET':np.float64, 'SP500':np.float64, 'turnover':np.float64}, header=0)
dsf.rename(columns={'turnover':'TURNOVER'}, inplace=True)

fillings = pd.read_csv('CUSIP.csv', header=0)
fillings.loc[:,'edate'] = [datetime.datetime.strptime(s, '%Y-%m-%d')  for s in fillings.loc[:,'edate']]
fillings.loc[:,'CAR0'] = np.nan
fillings.loc[:,'CAR1'] = np.nan
fillings.loc[:,'CAR2'] = np.nan
fillings.loc[:,'CAR3'] = np.nan
fillings.loc[:,'CAR5'] = np.nan
fillings.loc[:,'CAV0'] = np.nan
fillings.loc[:,'CAV1'] = np.nan
fillings.loc[:,'CAV2'] = np.nan
fillings.loc[:,'CAV3'] = np.nan
fillings.loc[:,'CAV5'] = np.nan

gspc = pd.read_csv('GSPC.csv', header=0)
gspc.loc[:,'Date'] = [int(s[0:4]+s[5:7]+s[8:10])  for s in gspc.loc[:,'Date']]
gspc.loc[:,'SP500'] = gspc.loc[:,'Adj Close'] / gspc.loc[:,'Adj Close'].shift(1) - 1
gspc = gspc.loc[:,['Date','SP500']]
gspc.dropna(axis=0, how='any', inplace=True)

dsf = pd.merge(dsf, gspc, how='inner', left_on='DATE', right_on='Date')
dsf = dsf.loc[:,['CUSIP','DATE','RET','SP500','TURNOVER']]
dsf.sort_values(by=['CUSIP','DATE'], ascending=[True,True], inplace=True)

reg_0 = 238
reg_1 = 63
to_0 = 49
to_1 = 8

for i in range(len(fillings)):
    print(i)
    
    T = dsf.loc[dsf.loc[:,'CUSIP']==fillings.loc[i,'cusip'],:]
    d = int(fillings.loc[i,'edate'].strftime('%Y%m%d'))
    p = np.where(T.loc[:,'DATE']==d)[0]
    if (len(p) == 0):
        continue
    if (len(p) > 1):
        print('err')
    p = p[0]
    if (p < reg_0  or  p < to_0):
        continue

    Y = T.loc[:,'RET'].values[p-reg_0:p-reg_1]
    X = T.loc[:,'SP500'].values[p-reg_0:p-reg_1]
    X = np.array([X]).transpose()
    X1 = st.add_constant(X)

    result = st.OLS(Y,X1).fit()

    mean_to = np.mean(T.loc[:,'TURNOVER'].values[p-to_0:p-to_1])
    std_to = np.std(T.loc[:,'TURNOVER'].values[p-to_0:p-to_1])

    for j in range(6):
        if (j == 4):
            continue
        if (p+j >= len(T)):
            continue
        S = T.iloc[p-j:p+j+1,:]
        fillings.loc[:,'CAR'+str(j)].values[i] = np.sum(S.loc[:,'RET'] - result.params[0] - result.params[1] * S.loc[:,'SP500'])
        fillings.loc[:,'CAV'+str(j)].values[i] = np.sum((S.loc[:,'TURNOVER'] - mean_to) / std_to)


fillings.dropna(axis=0, how='any', inplace=True)

mean = fillings.mean()
std = fillings.std()
med = fillings.median()
quantile1 = fillings.quantile(q=0.25, axis=0, numeric_only=True, interpolation='linear')
quantile2 = fillings.quantile(q=0.75, axis=0, numeric_only=True, interpolation='linear')

fillings_df = pd.concat([mean, std], axis=1)
fillings_df = pd.concat([fillings_df, med], axis=1)
fillings_df = pd.concat([fillings_df, quantile1], axis=1)
fillings_df = pd.concat([fillings_df, quantile2], axis=1)
fillings_df.columns = ['Mean','Std','Median', '25th Quantile','75th Quantile']
fillings_df = fillings_df[1:][:]
fillings_df.to_csv("Stats.csv", sep=',')

matplotlib.use('Agg')

fig = plt.figure(0, figsize=(10,6))
plt.subplot(231)

sns.distplot(fillings.loc[:,'CAR0'], hist=False, rug=False)
plt.subplot(232)

sns.distplot(fillings.loc[:,'CAR1'], hist=False, rug=False)
plt.subplot(233)

sns.distplot(fillings.loc[:,'CAR2'], hist=False, rug=False)
plt.subplot(234)

sns.distplot(fillings.loc[:,'CAR3'], hist=False, rug=False)
plt.subplot(236)

sns.distplot(fillings.loc[:,'CAR5'], hist=False, rug=False)
plt.savefig('CAR.jpg')
plt.close(0)

fig = plt.figure(0, figsize=(10,6))
plt.subplot(231)

sns.distplot(fillings.loc[:,'CAV0'], hist=False, rug=False)
plt.subplot(232)

sns.distplot(fillings.loc[:,'CAV1'], hist=False, rug=False)
plt.subplot(233)

sns.distplot(fillings.loc[:,'CAV2'], hist=False, rug=False)
plt.subplot(234)

sns.distplot(fillings.loc[:,'CAV3'], hist=False, rug=False)
plt.subplot(236)

sns.distplot(fillings.loc[:,'CAV5'], hist=False, rug=False)
plt.savefig('CAV.jpg')
plt.close(0)

