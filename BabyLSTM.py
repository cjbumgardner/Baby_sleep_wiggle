# -*- coding: utf-8 -*-

"""LSTM network for determining the probability that a baby is sleeping from accelerometer
data of 3D acceleration vectors recorded with an average frequency of 6Hz. 

Design 1: An two phase LTSM network. The first will look at a (outer) time window, T_o, of 15-30 seconds (TBD). 
The LSTM cell (cell size on the order of 10-30) will look at readings from every 1-5 seconds (inner 
time window T_i) within T_o. The inner LSTM cell will produce outputs chronologically partioning
the outer window, and the outputs to the inner LSTM cell will be inputs to the outer LSTM cell. 
The outer time window will slide along the entire time series sliding forward in time by 1-5 
seconds (thus overlapping windows much like a CNN window).
the 

"""

"""




tf.nn.softmax #https://www.tensorflow.org/api_docs/python/tf/nn/softmax

tf.nn.sigmoid_cross_entropy_with_logits  #https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits

x = tf.placeholder(tf.float32, shape=[None, 3,timeintervalTBD]) #None here for batchsize TBD

with tf.Session as sess:
    sess.run(graph, feed_dict={placeholder:blah,etc})

targetvalue = tf.placeholder(tf.float32, shape=[None,1]) # 1D for Awake=0 Asleep=1

https://www.tensorflow.org/programmers_guide/threading_and_queues

The above is for possibly helping with the "triangle" style LSTM where we need to 
write output of several rounds of the LSTM cell "sliding" through an interval, then use 
these as input to another LSTM cell. If nothing else is more authentic for this task...

NOTES:
Possible cell wrappers: tf.contrib.rnn.AttentionCellWrapper
This is similar to a kind of design thought of first where there is a RNN "window" that 
is used to observe outputs of a "sub"-RNN bi-directional structure. Not sure of the exact
structure of this wrapper funtion.



 """
import tensorflow as tf
import numpy as np
import threading
import Queue
from random import shuffle
import cPickle
from __future__ import division
#undefined variables: 
#files, capacity, num_reader_threads, batch_size, num_epochs=
class config(object):
    def __init__(self):
        #the settings for the LSTM bidirectional cells
        self.bidir_num_units=[60,60]
            #self.output_keep_prob=   for now leaving dropout out 
            #self.max_grad_norm=  leaving out any clipping with adam optimizer for now
        
        #for AdamOptimizer. 
        self.learning_rate=0.001
        self.beta1=0.9 #first moment decay rate for gradients
        self.beta2=0.99 #second moment decay rate for gradients
        self.epsilon=0.01
        
        #for queue runners and batch size for evaluating model 
        self.batch_size=10
        self.num_reader_threads=30#number of threads that are reading data from pickle into tf.tensor
        self.capacity=3*self.batch_size #capacity for the queue storing tf.tensors for batches
        self.sample_time_duration=30 # this the time duration of a single mark of sleep y/n. Most likely 15-60sec
        


#data files stored as dictionaries of np.ndarray {"data":shape=(None,sequence_length,time_chunk,3D),
#"labels":shape=(None,sequence_length,1),"sequence_length":shape=(None,1)}
def unpickle(filepath):
    with open(filepath,"rb") as obj:
        filey=cPickle.load(obj)
    return filey

class batcher(object):
    def __init__(self,files,config):
        self._files=files
        num_reader_threads=config.num_reader_threads
        capacity=config.capacity
        batch_size=config.batch_size
        sample_time=config.sample_time_duration
        
        self._data=data=tf.placeholder(dtype=tf.float32,shape=[1,None,sample_time])
        self._labels=labels=tf.placeholder(dtype=tf.float32,shape=[1,None,1])
        self._sequence_length=sequence_length=tf.placeholder(dtype=tf.int32,shape=[1,1])
        
        #queues and operations
        
        self._file_q=Queue.Queue(maxsize=len(files))
        self._batch_q=tf.PaddingFIFOQueue(capacity,(tf.float32,tf.float32),shapes=[[1,None,sample_time],[1,None,1],[1,1]],\
                    name="Batch_Queue")
        self._enqueue_op=self._batch_q.enqueue({"data":data,"labels":labels,"sequence_length":sequence_length},name="Batch_Enq_Op")
        self._output_op=self._batch_q.dequeue_many(batch_size,name="Batch_Deq_Op")

    def start_batch_runners(self,sess,coord):
        
        #python file thread. puts files into file_q and shuffles the files before enqueueing 
        def file_thread(coord,files,file_q):
            def filer(coord,files,file_queue):
                while not coord.should_stop():
                    shuffle(files)
                    for f in files:
                        if not coord.should_stop(): #put files into q
                            file_queue.put(f)
                        else:                     #if stopped during enq round then dump the q
                            while not file_queue.empty():
                                file_queue.get(block=False)
                                file_queue.task_done()
                            break
            file_thread=threading.Thread(target=filer,args=(coord,files,file_q),name="Filer_Thread")
            file_thread.daemon=True
            return file_thread


        #python unpickle reader threads
        def reader_threads(coord,sess,file_queue,enqueue,num_reader_threads):
            def reader():
                while not coord.should_stop():
                    x=file_queue.get()
                    file_queue.task_done()
                    x=unpickle(x) #dict of data,labels,sequence_length
                    sess.run(enqueue, feed_dict={data:x["data"],labels:x["labels"],sequence_length:x["sequence_length"]})
            reader_threads=[]
            for i in range(num_reader_threads):
                reader_thread=threading.Thread(target=reader,name="Reader_Thread{}".format(i))
                reader_thread.daemon=True   
                reader_threads.append(reader_thread)

            return reader_threads
        self._py_threads=[file_thread(coord,self._files,self._file_q)]+reader_threads(coord,sess,self._file_q,\
                                                                          self._enquque_op,self.num_reader_threads)
        for thread in self._py_threads:
            thread.start()
 
         
        return self._py_threads
    @property
    def batch(self):
        return self._output_op # returns op to produce dict of data,labels,sequence_length
        
        

class baby_rnn(object):
    def __init__(self,input_list,config,is_training):
        self.inputs=input_list["data"]
        self.labels=input_list["labels"]
        self.seq_length=input_list["sequence_length"]
        
        self.output_keep_prob=tf.placeholder(tf.float32)
        #the bidirectional LSTM stack
        bidir_cells={}
        for d in ["fw","bw"]:
            cells=[]
            i=1
            for num_units in config.bidir_num_units:
                cell=tf.nn.rnn_cell.LSTMCell(num_units=num_units,state_is_tuple=True)
                #cell=tf.nn.rnn_cell.DropoutWrapper(cell,output_keep_prob=self.output_keep_prob) as of now it seems
                #the dropoutwrapper in tf affects dropping memory states (7/17) which causes memory state exp growth
                #but also isn't in line with wanting to drop only output to next layer within timestep and not memory drop
                cells.append(cell)
                i+=1
            bidir_cells[d]=tf.nn.rnn_cell.MultiRNNCell(cells)
        outputs_bi,_=tf.nn.bidirectional_dynamic_rnn(cell_fw=bidir_cells["fw"],\
                cell_bw=bidir_cells["bw"],dtype=tf.float32,seqence_length=self.seq_length,inputs=self.inputs)
        num_outputs=config.bidir_num_units[-1]
        W=tf.get_variable("w_logit",[num_outputs],dtype=tf.float32)
        B=tf.get_variable("b_logit",[],dtype=tf.float)
        output_logits=tf.einsum('ijk,k->ij',outputs_bi,W)+B
        self._output_sigmoid=tf.sigmoid(output_logits)
        
        
        max_seq_length=max(self.seq_length)
        output_xtropy=tf.nn_sigmoid_cross_entropy_with_logits(logits=output_logits, labels=self.labels)
        mask=tf.convert_to_tensor([[1 for _ in range(self.seq_length[i])]+[0 for _ in range(max_seq_length-self.seq_length[i])] \
            for i in range(len(self.seq_length))],dtype=tf.float32)
        flat_output_xtropy=tf.reshape(output_xtropy,[-1])
        flat_mask=tf.reshape(mask,[-1])
        self._loss=loss=tf.reduce_mean(tf.boolean_mask(flat_output,flat_mask))
        if is_training==False:
            return
        #tvars=tf.trainable_variables()
        #clipping. doesn't necessarily make sense with ADAM which already normalizes the grad when used in steps
        #grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars),config.max_grad_norm)
        #train_step = tf.contrib.opt.ScipyOptimizerInterface(
               # loss,
               # method='L-BFGS-B',
                #options={'maxiter': iterations}) an implementation of inverse hessian optimization
                #as I can tell now, seems tf only has sgd methods. newton type methods can be pulled from scipy
        #adam optimizer variables                              
        lr=config.learning_rate
        b1=config.beta1
        b2=config.beta2
        e=config.epsilon
        
        optimizer=tf.train.AdamOptimizer(learning_rate=lr,beta1=b1,beta2=b2,epsilon=e)
        self._gradient=tf.placeholder(tf.float32)
        self._global_step=tf.Variable(0,name="global_step",trainable=False)
        self._train_op=optimizer.minimize(loss,grad_loss=self._gradient,global_step=self._global_step) #gate_gradients=GATE_OP, grad_loss=None place to store gradients if needed
        
        
    @property
    def train_op(self):
        return self._train_op
        
    @property
    def outputs(self):
        return self._output_sigmoid
    
    @property
    def loss(self):
        return self._loss
    
    @property
    def gradient(self):
        return self._gradient
    @property
    def global_step(self):
        return self._global_step
    

#def run_net(session,model,eval_op=None):
    
def variable_summaries(var_name,var):
    a=tf.summary.scalar(var_name,var)
    b=tf.summary.scalar(var_name+"_max",tf.reduce_max(var))
    c=tf.summary.scalar(var_name+"_l2norm",tf.norm(var))
    d=tf.summary.histogram(var_name+"_histogram",var)
    return a,b,c,d
        
    

    
def main():    
    train_validate_config=config()
    graph=tf.Graph()
    with graph.as_default():
        #could change range of uniform distribution
        initializer = tf.random_uniform_initializer(0,1)
       
        with tf.name_scope("Train"):
            train_batcher=batcher(train_files,train_validate_config)
            with tf.variable_scope("Model",initializer=initializer):
                train_model=baby_rnn(train_batcher.batch,config=train_validate_config,is_training=True) 
                t_l=tf.summary.scalar("train_loss",train_model.loss)
                a,b,c,d=variable_summaries("gradient",train_model.gradient)
        merged_train=tf.summary.merge([t_l,a,b,c,d])       
        with tf.name_scope("Validate"):
            validate_batcher=batcher(validate_files,train_validate_config)
            with tf.variable_scope("Model",reuse=True):
                validate_model=baby_rnn(validate_batcher.batch,config=train_validate_config)
                v_l=tf.summary.scalar("validate_loss",validate_model.loss)
        merged_validate=tf.summary.merge([v_l])    
        #will write code later for actually using the RNN for data analysis
        """with tf.name_scope("Test"):
            test_set=batcher(test_files,reader_threads=1,batch_size=1)
            with tf.variable_scope("Model",reuse=True):
                test_model=baby_rnn(test_set.batch,config= MISSING  )"""
        
        
        
        writer=tf.summary.FileWriter(MISSING path,graph=graph)
        init_op=tf.global_variable_initializer()  
        with tf.Session() as sess:
            sess.run(init_op)
            coord = tf.train.Coordinator()
            train_threads=train_batcher.start_batch_runners(sess,coord)
            validate_threads=validate_batcher.start_batch_runners(sess,coord)
            
            for i in range(num_epochs):
                if i%10==0:
                    loss,summary=sess.run(validate_model.loss,merged_validate)
                    print("The validation loss at epoch {}: {}".format(i,loss))
                    writer.add_summary(summary,i)
                else:
                    loss, summary=sess.run(train_model.loss,merged_train)
                    writer.add_summary(summary,i)
                                    
            test_tread=test_set.start_batch_runners(sess,coord)        
            #do test with final net settings        
    
            coord.request_stop()
            coord.join([train_threads,validate_threads,test_thread])
            
            
            writer.add_graph(sess.graph)
    


