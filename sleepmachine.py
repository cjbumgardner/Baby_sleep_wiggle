# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 19:13:46 2017

@author: Colorbaum
"""
from __future__ import division
import pandas as pd
import seaborn as sns
import numpy as np
import math
import matplotlib.pylab as plt
import os
"""An interactive program to help determine sleep=1, awake=0 for a baby's accelerometer data.
The following imports a csv file of accelerometer data, extracts a pandas db of time,x,y,z 
(where x,y,z are the accelerometer readings relative to chest x:left/right, y:up/down, z: forward/backward).
It makes a sleep column with the data initialized to 0.5.

It provides means to look at graphs, extend window, label graphs, and a help function for 
widening a window in time till the 3+1D acceleration vectors exhibit a user determined change 
in a quantity proportional to an approximation of a Hausdorff type dimension. 

Also, provides a function to convert and store data as a numpy file to be read into LSTM network.


"""
#converter for taking csv files and converting them into a dictionary={data=np.ndarray,labels=np.ndarray}
# to be fed into LSTM network. If they aren't for training, they will have data.shape=(#of minutes of file
#,60sec x 3 dim) and labels=empty. If they are for training, they will have data.shape=
#(max 60 min,180)

#reader for csv or xlsx files, returns pandas file with name=relavent file extension
def readbaby(filepath,mean=True,sleep=True,csv=False,xlsx=False):
    #args={mean=boolean,sleep=boolean}, mean to take average reading per second, default True
    if csv==True:
        db=pd.read_csv(filepath)
    if xlsx==True:
        db=pd.read_excel(filepath)
    if "_marked" not in filepath:
        ext=os.path.split(filepath)
        name=ext[1].split(".")[0]
        if sleep==True:
            filepath="/Users/Colorbaum/Desktop/DATADATDAT/Projects/BabyNOBaby/Data_Marked/"+name+"_marked.csv"
            
        else:
            filepath="/Users/Colorbaum/Desktop/DATADATDAT/Projects/BabyNOBaby/Data/Data_unmarked/"+name+"_unmarked.csv"
    #here marked identifies files that are hand marked and to be used for training/testing
    #unmarked identifies files that are in the pipeline for conversion for being marked by NNet
        
    if "sleep" not in db.columns:
        db=pd.DataFrame(db,columns=('Timestamp (client)','x','y','z'))
        db=db.rename(columns={"Timestamp (client)":"t"})
        if mean==True:
            db=db.groupby("t").mean()
            db=db.reset_index()
        if sleep==True:
            db['sleep']=0.5 
            for i in db.index:
                if i==0:
                    db.loc[i,"skip"]=-2
                else:
                    if db.t[i]-db.t[i-1]>30:
                        db.loc[i,"skip"]=1
                    else:
                        db.loc[i,"skip"]=-2
    db.name=filepath
    return db    #db=(t,x,y,z,sleepvalue,skip) if sleep=True, db=(t,x,y,z) if sleep=False

def picture(data,a,b):
    picx=sns.tsplot(data.iloc[a:b].x,time=data.iloc[a:b].index,color="red")
    axesx=picx.axes
    axesx.set(ylim=(-1.1,1.1))
    picy=sns.tsplot(data.iloc[a:b].y,time=data.iloc[a:b].index,color="green")
    axesy=picy.axes
    axesy.set(ylim=(-1.1,1.1))
    picz=sns.tsplot(data.iloc[a:b].z,time=data.iloc[a:b].index)
    axesz=picz.axes
    axesz.set(ylim=(-1.1,1.1))  
    picskip=sns.tsplot(data.iloc[a:b].skip,time=data.iloc[a:b].index,marker="v",color="black", interpolate=False)
    picskip.axes.set(ylim=(-1.1,1.1))
    title="Time interval = [{},{}]".format(a,b)
    plt.suptitle(title,fontsize=12,fontweight="bold")
    plt.show()
# compute a statistical estimate for an approx of Hausdorff-ish dim of a timeseries over a given interval
def selfsim(a,b,data):
    p=int(math.floor(math.log(b-a,2)))
    l=2**p
    df=data.iloc[a:a+l].to_matrix(columns=['x','y','z'])
    N=int(min(p, 10))
    array=np.zeros((N-1,2))
    for i in range(N-1):
        n=int(l/(2**i))
        array[i,0]=i-p # using log_2 this is log_2(1/r) where r is length 
        r=np.empty((2**i,3),dtype=np.float64)
        for k in range(2**i):
            r[k,:]=df[k*n:(k+1)*n].std(axis=0)
        xx=(r**2).sum(axis=1).reshape((2**i,1))
        array[i,1]=math.log(np.apply_along_axis(math.sqrt,arr=xx,axis=1).sum())
    return np.polyfit(array[:,0],array[:,1],1)[0]
        
    
def save(data):
    data.to_csv(data.name,index=False)    
    
    


class sleep(object):
    def __init__(self,db):# import pandas dataframe with columns t,x,y,z,sleep
        self.data=db
        self.marker=((db.sleep-0.5)**2).argmin()
        self.window=[self.marker,self.marker+60]
    def mark(self):
        command=""
        while True:
            print(self.window)
            picture(self.data,self.window[0],self.window[1])
            while command not in ("r","m","h","q","s"):
                command=raw_input("Type r: resize window right endpoint, m: mark data,\n  \
                h: help with finding endpoint, s: save, q: quit: ")
            if command=="s":
                save(self.data)
                command=""
                continue
            if command=="r":
                w=input("Number of seconds from window right endpoint: ")
                self.window=[self.window[0],self.window[1]+w]
                command=""
                continue
            if command=="m":
                m=input("Type 0:not sleeping, 1: sleeping: ")
                self.data.loc[self.window[0]:self.window[1],"sleep"]=m
                self.window=[self.window[1],self.window[1]+60]
                command=""
                continue
            if command=="q":
                if raw_input("Would you like to save? (y) or (n): ")!="n":
                    save(self.data)
                break
            if command=="h":
                center=selfsim(self.window[0],self.window[1],self.data)
                print("Initialize the features.")
                step=input("Input the step size in seconds (>2**10): ")
                tolerance=input("Input a tolerance (~10**-2): ")
                maxsteps=input("Input max number of steps: ")
                def run_hdim(rnd):    
                    count=0
                    hd=center
                    tol=0
                    leap=rnd*maxsteps*step
                    while abs(center-hd)<tolerance or count<maxsteps:
                        hd=selfsim(self.window[1]+step*count+leap,self.window[1]\
                        +step*(count+1)+leap,self.data)
                        count+=1
                    else:
                        if abs(center-hd)>=tolerance:
                            tol=1   
                    picture(self.data,self.window[0],self.window[1]+step*count+leap)
                    picture(self.data,self.window[1]+step*(count-1)+leap,self.window[1]+step*count+leap)
                    return self.window[0],self.window[1]+step*count,tol,count
                rnd=0
                while True:
                    left,right,tol,count=run_hdim(rnd)
                    if tol==0:
                        c=raw_input("The tolerance wasn't exceeded. Would you like\
                        to run again? (y) or (n): ")
                        if c=="y":
                            rnd+=1
                            continue
                    else: 
                        print("The tolerance was exceeded after {} steps".format(rnd*maxsteps+count))
                        break
                
                c=raw_input("Type (x) to exit help and not keep window, (k) to keep window,\n \
                (ks) to keep window except last step: ")
                if c=="k":
                    self.window=[self.window[0],self.window[1]+step*(count+rnd*maxsteps)]
                if c=="ks":
                    self.window=[self.window[0],self.window[1]+step*(count-1+rnd*maxsteps)]
                command=""
                continue        
                        
                        
              

   
        
