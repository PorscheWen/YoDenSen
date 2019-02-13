# -*- coding: utf-8 -*-
"""
Created on Mon Dec 31 14:35:01 2018

@author: User
"""

import pandas as pd
from finlab.YDS_backtest import backtest
from finlab.data import Data
import datetime


# Remove 'Fututre Warning'
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def PickDate(strYMD):
    # Input strYMD = '2018_01_31'
    dateYMDT = datetime.datetime.strptime(strYMD, '%Y_%m_%d').replace(tzinfo = None)
    dateYMD = dateYMDT.replace(tzinfo = None)
    return dateYMD


def dic_ROE(equality, RSV_Ndays, StockHold_Days):
    SumROEPercent = (equality[-1]-1)*100
    SumROEPercent_Round = round(SumROEPercent, 2)
    #print('總報酬率 : '+str(SumROEPercent_Round)+'%')    
    #dicData = {'RSV_Ndays': RSV_Ndays, 'StockHold_Days': StockHold_Days, 'SumROE': SumROEPercent_Round}
    dicData = {'RSV_Ndays': RSV_Ndays, 'StockHold_Days': StockHold_Days, 'SumROE': SumROEPercent_Round}
    #SumROE_DF = pd.DataFrame(dicData)
    return dicData


if __name__ == '__main__':
    SumROE_DF_All = pd.DataFrame()
    for RSV_Ndays in range (50, 201, 50):
        for StockHold_Days in range (30, 61, 30):
            # Input:
            StartDate = PickDate('2015_01_01')
            EndDate = PickDate('2018_12_31')
            print('RSV_Ndays: ' + str(RSV_Ndays))
            print('StockHold_Days: ' + str(StockHold_Days))
            
            # Evaluation:
            # equality是累計總報酬率  2018-11-30    0.859861
            # transections是買賣個股的價格 1701     2018-05-03      20.90     ...     2018-05-31       20.55
            equality, transections = backtest(StartDate, EndDate, StockHold_Days, RSV_Ndays, Data())
           
            
            # Output:
            dicData = dic_ROE(equality, RSV_Ndays, StockHold_Days)
            SumROE_DF_All = SumROE_DF_All.append(dicData, ignore_index = True)
    SumROE_DF_All_Sort = SumROE_DF_All.sort_values( by = 'SumROE',  ascending= True)
    print(SumROE_DF_All_Sort)
    
'''   
    # 測試最佳 RSV_Ndays 50 - 200天
    for RSV_Ndays in range(50 , 201,  50):
        for Sold_NDays in range (30, 61,30):
            print('RSV '+str(RSV_Ndays)+' days')
            print('持股天數 '+str(Sold_NDays)+' days')
            data = Data()  
            stocks = mystrategy2(data, RSV_Ndays)
            #equality是累計總報酬率  2018-11-30    0.859861
            #transections是買賣個股的價格 1701     2018-05-03      20.90     ...     2018-05-31       20.55
            equality, transections, SumROE = backtest(datetime.date(2018,1,1), datetime.date(2018,12,1), Sold_NDays, stocks, data)
            #SumROE = round(equality[-1],4)*100-100
            RSV_DF = pd.DataFrame({'RSV_NDays': [RSV_Ndays], '持股天數':[Sold_NDays], '總報酬率':[SumROE]})
    print(RSV_DF)
        
        
         
    # 回測
    #backtest(datetime.date(2018,5,1), datetime.date(2018,12,1), 30, mystrategy2, data)
    
  
    # 利用 mystrategy2 來產生股票清單 stocks
    stocks = mystrategy2(data)
    
    # 用portfolio來幫忙計算，給定 300,000 元，依照今天收盤價，股票張數要如何分配
    # 印出股票資訊
    InvestAmount = 300000
    p, total_invest_money = portfolio(stocks.index, InvestAmount, data)
    
    print('-'*30)
    print('投入 '+ str(InvestAmount) +' \n如何分配股票張數')
    print('-'*30)
    print(p)
    print('total cost: '+str(total_invest_money))    
'''