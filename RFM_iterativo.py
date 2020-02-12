# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 10:06:19 2020

@author: dlopezmacias
"""


import pandas as pd
import progressbar
from dateutil.relativedelta import relativedelta
from datetime import timedelta
import datetime
import os

pd.options.mode.chained_assignment = None

# Categorise
    
    ## for Recency 
    
def RClass(x,p,d):
    if x <= d[p][0.25]:
        return 1
    elif x <= d[p][0.50]:
        return 2
    elif x <= d[p][0.75]: 
        return 3
    else:
        return 4
    
## for Frequency and Monetary value 

def FMClass(x,p,d):
    if x <= d[p][0.25]:
        return 4
    elif x <= d[p][0.50]:
        return 3
    elif x <= d[p][0.75]: 
        return 2
    else:
        return 1   
    
def RFM_nimerya_iterativo(filename,date_column,idcustomer,sales_column):
    
    df = pd.read_csv(filename,sep=';')
    
    print('Working...')
    
    # trasnform
           
    df[date_column] = pd.to_datetime(df[date_column]).dt.date
    
    df = df[pd.notnull(df[idcustomer])]
      
    # Calculate the period id 
    
    df['period_id'] = df[date_column].map(lambda x: 100*x.year + x.month)
    
    # We will only consider transactions from the last year
    
    # df=df[df['total_days'] < 365]
    
    """
    The data will be summarized at customer level by taking number of days to the 
    latest transaction, sum of all transction amount and total number of 
    transaction.
    
    """
    
    # Subset by period id 
    bar = progressbar.ProgressBar(maxval=len(df['period_id'].unique())).start()
    j = 0     
    
    df_list = list()
    
    df_final = pd.DataFrame()
    
    for i in df['period_id'].unique():
        j = j + 1 
        bar.update(j)
        
        # Subset df by periodid
        globals()['df_%s' % i] = df.loc[df['period_id'] == i]
         
        max_date = globals()['df_%s' % i][date_column].max()
         
        min_date = max_date - relativedelta(years=1)
        # Subset by date 
        
        temp_df = df.loc[(df[date_column] > min_date) & (df[date_column] <= max_date)]
        
        # globals()['df_%s' % i] = globals()['df_%s' % i].append(temp_df,sort = True)
        
        globals()['df_%s' % i] = pd.concat([globals()['df_%s' % i], temp_df], ignore_index=True, sort =False)
        
        # Calculate de Recency
        
        sd = globals()['df_%s' % i][date_column].max() + timedelta(days=1)
        
        # sd = sd - relativedelta(years=1)
        
        """
        We will calculate the recency based on a 1 year period, for this, we obtain
        the max date, and subtract 1 year.
        """
        
        globals()['df_%s' % i]['total_days'] = sd - globals()['df_%s' % i][date_column] 
    
        globals()['df_%s' % i]['total_days'].astype('timedelta64[D]')
    
        globals()['df_%s' % i]['total_days'] = globals()['df_%s' % i]['total_days'] / pd.np.timedelta64(1, 'D')
           
        # Group by user to calculate RFM
        globals()['rfmTable_%s' % i] = globals()['df_%s' % i].groupby(idcustomer).agg({'total_days': lambda x:x.max(), # Recency
                                            idcustomer: lambda x: len(x),  # Frequency
                                            'tran_amount': lambda x: x.sum()})      # Monetary Value
        globals()['rfmTable_%s' % i].rename(columns={'total_days': 'recency', 
                                                     idcustomer: 'frequency', 
                                                     'tran_amount': 'monetary_value'}, inplace=True)
        #Define the quartiles
        quartiles = globals()['rfmTable_%s' % i].quantile(q=[0.25,0.50,0.75])
     
        """
        let's convert quartile information into a dictionary so that cutoffs can be
        picked up.
        
        """
        quartiles = quartiles.to_dict()
        rfmSeg = globals()['rfmTable_%s' % i]
        rfmSeg['R_Quartile'] = rfmSeg['recency'].apply(RClass, args=('recency',quartiles,))
        rfmSeg['F_Quartile'] = rfmSeg['frequency'].apply(FMClass, args=('frequency',quartiles,))
        rfmSeg['M_Quartile'] = rfmSeg['monetary_value'].apply(FMClass, args=('monetary_value',quartiles,))
        
        """
        Now that we have the assigned numbers, let's use K-means to determine 
        the number of clusters 
        
        """
        from sklearn.cluster import KMeans
        
        cluster = KMeans(n_clusters=4)
        
        # slice matrix so we only include the 0/1 indicator columns in the clustering
        
        rfmSeg['cluster'] = cluster.fit_predict(rfmSeg[rfmSeg.columns[2:]])
        
        rfmSeg.cluster.value_counts()
        
        """
        As we can see there is not a clear distinction for the members so we will use
        another segmentation technique
        
        """
        rfmSeg['Total Score'] = rfmSeg['R_Quartile'] + rfmSeg['F_Quartile'] +rfmSeg['M_Quartile']
           
        rfmSeg.groupby('Total Score').agg('monetary_value').mean()
           
        # Rescale values from 1 - 10 
        
        rfmSeg['Total Score'] = pd.np.interp(rfmSeg['Total Score'], 
              (rfmSeg['Total Score'].min(), 
                rfmSeg['Total Score'].max()), (1, 10))
        
        rfmSeg.reset_index(idcustomer, inplace=True)
        
        globals()['rfmSeg_%s' % i] = rfmSeg
        
        """ 
            
        Now that we have structured the different customers into groups let's see if 
        distribution is validated by the pareto distribution
        
        """
        
        globals()['rfmSeg_%s' % i]['period_id'] = i
        
             
        # Generate a output pdf 
        df_final = df_final.append(globals()['rfmSeg_%s' % i])
        
        
    bar.finish()
    
    df_final.to_csv('rfm_'+str(filename), sep='|',index=False)
    
    directory = os.getcwd()
       
    print('File:', filename, 'generated in:', directory)