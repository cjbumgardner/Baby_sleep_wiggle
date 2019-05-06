"""Graphing and labeling tool with helper functions for time series sleep data. 

A tool to aid in labeling sleep=1/awake=0 timeseries data of 3D accelerometer 
reading that is intended to be used alongside sleep data estimates. Mostly
intended to be used with an interactive python notebook (like jupyter), it
provides functions to graph data for a user's given time window, label data 
within specified time intervals, and provides a helper function that looks the 
next point in time where the ``wiggle" in the current window changes outside a 
giventolerance. The latter function looks for changes in a statistical type of
Hausdorf dimension. 
"""


from __future__ import division
import os
import pandas as pd
import numpy as np
import math
import matplotlib.pylab as plt
import seaborn as sns


def readbaby(filepath, mean=False, sleep=False):
    """Reader for raw accelerometer data outputting augmented pandas df. 
    
    Raw acclerometer data with columns = ('Timestamp (client)', 'x', 'y', 'z')
    
    
    Args:
        filepath: csv or xlsx filepath 
        mean: Boolean. A value of True will group rows by timestamp value and 
            replace accelerometer readings with their mean.
        sleep: Boolean. A value of True will add a column "sleep" with value 0.5
            as well as a integer valued column "skip" to note if a 30 second or 
            more gap in timestamp data occured. 
        
    Returns: Pangas dataframe with columns = (t, x, y, z, sleepvalue, skip) if 
    sleep = True, or columns = (t ,x ,y ,z) if sleep = False. It's name is a 
    new filepath with csv extension. 
        
    Raises: ValueError if not a csv/xlsx filepath extension.
        
    """
    
    try:
        ext = os.path.splitext(filepath)[1]
        
        if ext == "csv":
            db = pd.read_csv(filepath)
        elif ext == "xlsx":
            db = pd.read_excel(filepath)
        else:
            raise ValueError
        if "_marked" not in filepath:
            head, tail = os.path.split(filepath)
            name = tail.split(".")[0]
            if sleep == True:
                new_filepath = head + name + "_marked.csv"
            else:
                new_filepath = head + name + "_unmarked.csv"
            db.name = new_filepath
        else:
            db.name = filepath #if 'marked' is in name, then filepath ext = .csv
    except ValueError:
        print("File extension not csv or xlsx.")

    if "sleep" not in db.columns:
        try:
            db = pd.DataFrame(db, columns=('Timestamp (client)', 'x', 'y', 'z'))
            db = db.rename(columns={"Timestamp (client)": "t"})
            if mean == True:
                db = db.groupby("t").mean()
                db = db.reset_index()
            if sleep == True:
                db['sleep'] = 0.5
                for i in db.index:
                    if i == 0:
                        db.loc[0, "skip"] = -2
                    else:
                        if db.t[i] - db.t[i - 1] > 30:
                            db.loc[i, "skip"] = 1
                        else:
                            db.loc[i, "skip"] = -2
    return db  


def picture(data, a, b):
    """Plots accelerometer data for specified time window [a,b]. 
    
    Args: 
        data: pandas dataframe with columns = (t, x, y, z, sleepvalue, skip)
        a: unix timestamp beginning time
        b: unix timestamp end time
    
    Returns: 
        Four plots of x, y, z, and "skip" timeseries data. 
        
        """
    picx = sns.tsplot(data.iloc[a:b].x, time=data.iloc[a:b].index, color="red")
    axesx = picx.axes
    axesx.set(ylim=(-1.1, 1.1))
    
    picy = sns.tsplot(data.iloc[a:b].y,
                      time=data.iloc[a:b].index,
                      color="green")
    axesy = picy.axes
    axesy.set(ylim=(-1.1, 1.1))
    
    picz = sns.tsplot(data.iloc[a:b].z, time=data.iloc[a:b].index)
    axesz = picz.axes
    axesz.set(ylim=(-1.1, 1.1))
    
    picskip = sns.tsplot(data.iloc[a:b].skip,
                         time=data.iloc[a:b].index,
                         marker="v",
                         color="black",
                         interpolate=False)
    picskip.axes.set(ylim=(-1.1, 1.1))
    title = "Time interval = [{},{}]".format(a, b)
    plt.suptitle(title, fontsize=12, fontweight="bold")
    plt.show()


def selfsim(a, b, data):
    """Computes a statistical fractal dim of a timeseries over a given interval.
    
    Similar to other computations of fractal dimensions involving covering 
    schemes (e.g. with spheres), this is based roughly on a covering scheme with
    4D cylindars of decreasing heights (in time direction). Only the variance of
    the data is used to comute covering spheres for contracted time intervals.
    Since only achange in this quantity is useful, many constants to make a 
    'correct' computation of such a fractal dimension have been surpressed. 
    Note: This is a 'sloppy' but functional definition of a 'fractal dimension'
    for the intended purposes.
   
    Args: 
        a: unix timestamp for start time
        b: unix timestamp for end time
        data: pandas df with (x ,y ,z) in columns 
    
    Returns: 
        Float. Statistical type of fractal dimension.
    """
    p = int(math.floor(math.log(b - a, 2)))
    l = 2**p
    df = data.iloc[a:a + l].to_matrix(columns=['x', 'y', 'z'])
    N = int(min(p, 10))
    array = np.zeros((N - 1, 2))
    for i in range(N - 1):
        n = int(l / (2**i))
        array[i, 0] = i - p  #this is log_2(1/r) where r is length
        r = np.empty((2**i, 3), dtype=np.float64)
        for k in range(2**i):
            r[k, :] = df[k * n:(k + 1) * n].var(axis=0)
        xx = r.sum(axis=1).reshape((2**i, 1))
        array[i, 1] = math.log(
            np.apply_along_axis(math.sqrt, arr=xx, axis=1).sum())
    return np.polyfit(array[:, 0], array[:, 1], 1)[0]


def save(data):
    """Save function alias.
    
    Args: 
        data: pandas dataframe
        
    Returns: 
        None
    """
    data.to_csv(data.name, index=False)


class sleep(object):
    """Crude user interface for labeling accelerometer data. 
    
    The UI begins by displaying a view of the data and gives the viewer options
    to label the data and move to the next window, enlarge/shrink the window, 
    find largest window with selfsim() output within a tolerance interval, save,
    and quit labeling. It is made so that labeling a single file can be done 
    after saving and continuing. 
    
    Attributes:
        db: pandas dataframe with columns (t,x,y,z,sleep,skip)
        marker: smallest index where db has sleep not yet marked 0/1. 
        window: time interval for viewing data
    """

    def __init__(self, db):
        """Initializing a marker and window. 
        
        Args: 
            db = pandas dataframe
        """
        self.data = db
        self.marker = ((db.sleep - 0.5)**2).argmin()
        self.window = [self.marker, self.marker + 60]

    def mark(self):
        """Runs the main labeling interface."""
       
        command = ""
        while True:
            print(self.window)
            picture(self.data, self.window[0], self.window[1])
            while command not in ("r", "m", "h", "q", "s"):
                command = raw_input(
                    "Type r: resize window right endpoint, m: mark data,\n  \
                h: help with finding endpoint, s: save, q: quit: ")
            if command == "s":
                save(self.data)
                command = ""
                continue
            if command == "r":
                w = input("Number of seconds from window right endpoint: ")
                self.window = [self.window[0], self.window[1] + w]
                command = ""
                continue
            if command == "m":
                m = input("Type 0:not sleeping, 1: sleeping: ")
                self.data.loc[self.window[0]:self.window[1], "sleep"] = m
                self.window = [self.window[1], self.window[1] + 60]
                command = ""
                continue
            if command == "q":
                if raw_input("Would you like to save? (y) or (n): ") != "n":
                    save(self.data)
                break
            if command == "h":
                center = selfsim(self.window[0], self.window[1], self.data)
                print("Initialize the features.")
                step = input("Input the step size (>2**10): ")
                tolerance = input("Input a tolerance (~10**-2): ")
                maxsteps = input("Input max number of steps: ")

                def run_hdim(rnd):
                    count = 0
                    hd = center
                    tol = 0
                    leap = rnd * maxsteps * step
                    while abs(center - hd) < tolerance or count < maxsteps:
                        hd=selfsim(self.window[1]+step*count+leap,self.window[1]\
                        +step*(count+1)+leap,self.data)
                        count += 1
                    else:
                        if abs(center - hd) >= tolerance:
                            tol = 1
                    picture(self.data, self.window[0],
                            self.window[1] + step * count + leap)
                    picture(self.data,
                            self.window[1] + step * (count - 1) + leap,
                            self.window[1] + step * count + leap)
                    return (self.window[0], self.window[1] + step * count, tol,
                            count)
                      
                rnd = 0
                 
                while True:
                    left, right, tol, count = run_hdim(rnd)
                    if tol == 0:
                        c = raw_input(
                            "The tolerance wasn't exceeded. Would you like\
                            to run again? (y) or (n): "
                            )
                        if c == "y":
                            rnd += 1
                            continue
                    else:
                        print(
                            "The tolerance was exceeded after {} steps".format(
                                rnd * maxsteps + count))
                        break

                c = raw_input(
                    "Type (x) to exit help and not keep window, (k) to keep \
                    window,\n (ks) to keep window except last step: "
                    )
                if c == "k":
                    self.window = [
                        self.window[0],
                        self.window[1] + step * (count + rnd * maxsteps)
                    ]
                if c == "ks":
                    self.window = [
                        self.window[0],
                        self.window[1] + step * (count - 1 + rnd * maxsteps)
                    ]
                command = ""
                continue

def main():
    """For using labeling program on the command line.

    """
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",
                        "-d", 
                        help="The filename for data to be labeled",
                        )
    parser.add_argument("--mean", "-m", action= "store_true", 
                        help="Option used for groupby timestamp then take mean")
    parser.add_argument("--sleep","-s", action="store_true",
                        help="Option for initialize sleep column to 0.5")
    args = parser.parse_args()
    filepath = args.data
    mean = args.mean
    sleep = args.sleep
    df = readbaby(filepath, mean=mean, sleep=sleep)
    sl = sleep(df)
    sl.mark()
    
if __name__ == "__main__":
    main()