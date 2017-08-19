# Baby_sleep_wiggle
RNN and labeling tool for determining sleep habits of infants from accelerometer data.

The sleepmachine.py is a primative graphical interface to label data as sleeping or not sleeping. Amongst some tools to choose
window size for the timeseries data, label, and store timeseries data (with cpickle), it also contains a helper function that computes the largest window (forward in time) where 
there is no change (relative to a user defined interval) in a quantity that is proportional to a Hausdorff-like dimension. 
Basically, it helps you find a largest window where the "wiggle" of the baby is roughly the same as that of a chosen segment of 
time. 

The BabyLSTM.py is a training program for a bidirectional recurrent neural network using LSTM cells. It uses
Tensorflow. It assumes the input data is stored in a cpickle format. 

This program is for a larger research project for the Nutional Science Department at UC Davis. 
