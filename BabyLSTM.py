"""
@author: Christopher Bumgardner
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from typing import Optional, List, Text, Dict
import math
from pathlib import Path
import tensorflow as tf
import numpy as np


"""A 'wedding cake' style RNN network classifying subsequence information.

A RNN network for reading sequential information and outputting a classification
or probability variable for sequential subsequences. For example, if one took  
accelerometer readings from a wearable device at the rate of 1 Hz and wished to
determine the probability the wearer was either running, sitting, or sleeping
over each one minute interval. 

The design has several tunable parameters for the network geometry. The 'tiers'
to the 'wedding cake' are standard multilayered RNN networks with variable
depth and number of units in each layer. Before each tier, there are options 
for a CNN style window and stride over the timeseries. Meaning, several 
sequential units of time (the window) can be read by single RNN cells, and then
the window moves through the sequence by a specified step size (the stride). 
Again, much like convolutions in CNNs, the sequence length will shorten after 
each tier.

There are options for either sequential categorical output or simply one 
categorical output. This is specified by adding a dense layer or not in a 
params class. 

In the params class, the network geometry is specified by the following.
tiers = {0:[rnn cell size in 1st layer, 2nd layer],
        1:[rnn cell size 1st layer, 2nd layer, etc],etc}
window_size = {0: window size for before being fed into 1st rnn tier,
               1:window size for before being fed into 2nd rnn tier,etc}
stride = {0: stride for before being fed into 1st rnn tier,
          1: stride for before being fed into 1st rnn tier, etc} 

These are parameters for a reshaping of the data, like convolutions, for feeding 
into a standard mulitlayered RNN (lstm or gru). They should be specified before 
each tier, so len(stride)=len(window_size)=len(tiers).
"""


class params(object):
    """Parameters for RNN network design, training, and logging.
    
    Attributes: 
        cell_type: string "lstm" or "gru", default "lstm"
        tiers: list of lists of ints [[int,],[int,],] (see module docs)
        window_size: list of ints (see module docs), default 1
        strides: list of ints (see module docs), default 1
        use_dense_layer_output: Boolean, Default False. This is for sequence to 
            category models where only one value is categorized, or a 
            probability distribution over categories. 
        dense_layer_args: optional dense layer args including number of units
        adam_opt_args: Dict of values for tf.adamoptimizer
        lstm_cell_args: Dict of values for tf.keras.layers.LSTMCell not 
            including number of units
        gru_cell_args: Dict of value for tf.keras.layers.GRUCell not including
            number of units
        tens_to_log: Dict of named tensors for logging and viewing during 
            training
        checkpoint_path: filepath where model checkpoints will be stored during
            training
        predict_batch_size: batch size when using the estimator predictor, 
            possibly only at the end of training. Note: this has nothing to do 
            with function 'cake_predict'
        save_checkpoints_steps: how many trainging steps before saving the next
            checkpoint,
        keep_checkpoint_max: max number of checkpoints to keep
        save_summary_steps: number of training steps before saving next summary
            for tensorboard usage
        logging_step: number of steps before next command line output of tens_to_log
    """
    def __init__(self,
                 cell_type: Text = "lstm",
                 tiers: Dict[int:List[int]] = {1:[1]},
                 window_sizes: List[int] = [1],
                 strides: List[int] = [1],
                 use_dense_layer_output : bool = False,
                 dense_layer_args : Dict = {"units": None,
                                           "activation":None,#don't change this
                                           "use_bias":True,
                                           "kernel_initializer":'glorot_uniform',
                                           "bias_initializer":'zeros',
                                           "kernel_regularizer":None,
                                           "bias_regularizer":None,
                                           "activity_regularizer":None,
                                           "kernel_constraint":None,
                                           "bias_constraint":None,
                                           },
                 lstm_cell_args : Dict = {"activation":'tanh',
                                        "recurrent_activation":'hard_sigmoid',
                                        "use_bias":True,
                                        "kernel_initializer":'glorot_uniform',
                                        "recurrent_initializer":'orthogonal',
                                        "bias_initializer":'zeros',
                                        "unit_forget_bias":True,
                                        "kernel_regularizer":None,
                                        "recurrent_regularizer":None,
                                        "bias_regularizer":None,
                                        "kernel_constraint":None,
                                        "recurrent_constraint":None,
                                        "bias_constraint":None,
                                        "implementation":1,
                                        },
                 gru_cell_args : Dict = {"activation":'tanh',
                                        "recurrent_activation":'hard_sigmoid',
                                        "use_bias":True,
                                        "kernel_initializer":'glorot_uniform',
                                        "recurrent_initializer":'orthogonal',
                                        "bias_initializer":'zeros',
                                        "kernel_regularizer":None,
                                        "recurrent_regularizer":None,
                                        "bias_regularizer":None,
                                        "kernel_constraint":None,
                                        "recurrent_constraint":None,
                                        "bias_constraint":None,
                                        "implementation":1,
                                        "reset_after":False,
                                        },
                 adam_opt_args : Dict = {"learning_rate": None,
                                         "epsilon": None,
                                         "beta_1": None,
                                         "beta_2": None,
                                         },
                 clip_norm : Optional[float] = None,
                 dropout_training = 0,
                 recurrent_dropout_training = 0,
                 tens_to_log = None,
                 checkpoint_path = None,
                 predict_batch_size = 30,
                 save_checkpoints_steps = 100,
                 keep_checkpoint_max = 10,
                 save_summary_steps = 10,
                 logging_step = 10,
                 ):
        """Initialize network geometry and training parameters.
        
        Raises: ValueError if there is a mismatch in the network geometry specs.
        For each tier, the network needs a specification of window size and 
        stride, so the list lengths specifying these need to all be the same.
        
        """
        
        if (len(tiers)!=len(window_sizes) or len(window_sizes)!=len(strides)):
           raise ValueError("Mismatch in network geometry specs. Need number of"
                            " tiers = number of window size specs, or number of"
                            " window specs = number of stride specs.")
        
        
        self.cell_type = cell_type 
        self.tiers = tiers, 
        self.window_sizes = window_sizes,
        self.strides = strides,
        self.use_dense_layer_output = use_dense_layer_output
        self.dense_layer_args = dense_layer_args,
        self.lstm_cell_args = lstm_cell_args,
        self.gru_cell_args = gru_cell_args,
        self.adam_opt_args = adam_opt_args,
        self.clip_norm = clip_norm,
        self.dropout_training = 0,
        self.recurrent_dropout_training = recurrent_dropout_training,
        self.tens_to_log = tens_to_log,
        self.checkpoint_path = None,
        self.predict_batch_size = 30,
        self.save_checkpoints_steps = 100,
        self.keep_checkpoint_max = 10,
        self.save_summary_steps = 20,
        self.logging_step = 10
        
    
def time_conv_reshape(arr,window,stride):
    """Reshape the sequence data for a convolutional style network.
    
    Converts the sequence 'arr' to another sequence by flattening features in
    time windows of size = window for every window beginning at a time index 
    that is a multiple of the value 'stride'. 
    
    Args: 
        arr: tf.tensor of shape [batch_size, time_steps, features]
        window: int window size 
        stride: int stride length
    Returns: 
        tf.tensor of shape [batch_size, new_time_steps, features*window]. Here
        new_time_steps = n + 1 (for n below)
    """
    
    bat, steps, feat = arr.get_shape().as_list()
    r = tf.floormod((steps - window), stride)
    n = math.ceil((steps - window)/stride)
    
    def padder(n=n,r=r,feat=feat,steps=steps,bat=bat,arr=arr):
        """Pad function."""
        pad = tf.zeros([bat, stride - r, feat],tf.float32)
        return tf.concat([arr, pad], 1) 
     
    arr = tf.cond(tf.equal(r,0), lambda: arr, padder)
    steps = tf.cond(tf.equal(r,0), lambda: steps, lambda: steps + stride -r)
    last_step = steps - window + 1 
    
    def c(i,a,b):
        """Condition tf.while_loop"""
        return tf.less(i,window)
    
    def b(i,new_arr,arr):
        """Body tf.while_loop. Appends ith value of windows to new_arr."""
        new_arr = tf.concat([new_arr,arr[:, i:last_step + i:stride, :]], axis=2)
        return i+1,new_arr,arr
    
    i = tf.constant(1)
    new_arr = arr[:, 0: last_step: stride, :]
    new_arr.set_shape([bat,n+1,None])
    _,new_arr,_=tf.while_loop(c,
                              b,
                              loop_vars=[i,new_arr,arr],
                              shape_invariants=[i.get_shape(),
                                                tf.TensorShape([bat,n+1,None]),
                                                arr.get_shape(),
                                                ],
                              )
    new_arr.set_shape([bat,n+1,feat*window])
    return new_arr  

     
def rnn_stack(params=None,
              tier : int = None,
              last_tier = False,
              dropout = 0,
              recurrent_dropout = 0,
              ):
    """MultiRnn cell. Options
    
    Args: 
        params = params class with cell args
        tier = int designating the tier number
        """
    layers = params.tiers[tier]
    def memory_cell(num_cells,last_layer=False):
        """"""
        if params.cell_type == "gru":
            kwargs = params.gru_cell_args
            if last_layer == True: 
                kwargs["activation"] = None
            cell = tf.keras.layers.GRUCell(num_cells,
                                           dropout = dropout,
                                           recurrent_dropout = recurrent_dropout,
                                           **kwargs,
                                           )
        if params.cell_type == "lstm":
            kwargs = params.lstm_cell_args
            if last_layer == True: 
                kwargs["activation"] = None
            cell = tf.keras.layers.LSTMCell(num_cells,
                                            dropout = dropout,
                                            recurrent_dropout = recurrent_dropout,
                                            **kwargs)
        return cell 
    
    if last_tier == False:
        cells = [memory_cell(l) for l in layers]
    else:
        cells = [memory_cell(l) for l in layers[:-1]]
        cells.append[memory_cell(layers[-1],last_layer=True)]#output is logits
    cell_stack = tf.keras.layers.RNN(cells,return_sequences=True)
    return cell_stack


def cake_fn(features,labels,mode,params):
    """Main 'wedding cake' style network function.
    
    Args:
        features: dict of features to be fed into model
        labels: tf.tensor of labels
        mode: tf.estimator.ModeKeys 
        params: dict of parameters for model
    """
    
    x = features["x"]
    tiers = params.tiers
    window_sizes = params.window_sizes
    strides = params.strides
    
    if mode == tf.estimator.ModeKeys.TRAIN:
        dropout = params.dropout_training
        recurrent_dropout  = params.recurrent_dropout_training
    else: 
        dropout = 0
        recurrent_dropout = 0
      
    for i in range(len(tiers)):
        x = time_conv_reshape(x, window_sizes[i], strides[i])
        stack = rnn_stack(params=params,
                          tier=i, 
                          dropout = dropout,
                          recurrent_dropout = recurrent_dropout,
                          )
        x = stack(x)
        
    if params.use_dense_layer_output == True:
        dense = tf.keras.layers.Dense(**params.dense_layer_args)
        logits = dense(x) 
    else: 
        logits = x 
        
    def loss_fn(y_true,y_pred):
        """Xentropy loss function for sequence output or category output. 
        
        Args:
            y_true: one hot true label
            y_pred: logit output of network
        """ 
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(y_true,
                                                          y_pred,
                                                          axis=-1,
                                                         )
        loss = tf.reduce_mean(loss,name="loss")
        return loss
    
    depth = logits.get_shape().as_list()[-1]
    predictions = {"probabilities": tf.nn.softmax(logits,
                                                  axis=-1,
                                                  name="probabilities",
                                                  ),
                  "labels":tf.one_hot(tf.argmax(logits,axis=-1),
                                  depth,axis=-1,
                                  name="output_labels",
                                  ),
                  }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode = mode,
                                          predictions = predictions,
                                          )

    loss = loss_fn(labels,logits)
    reg_loss = tf.losses.get_regularization_losses()
    loss = loss + tf.reduce_sum(reg_loss)
    acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(predictions["labels"],
                                                    axis=-1),
                                          tf.argmax(labels,axis=-1)),
                                 tf.float32),
                         name="accuracy_on_average",
                         )
    tf.summary.scalar("average_accuracy",acc)
    tf.summary.scalar("loss_with_regularization",loss)
   
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(**params.adam_opt_args)
        grads,varis = [*zip(*optimizer.compute_gradients(loss=loss))]
        if params.clip_norm != None:
            grads = [tf.clip_by_average_norm(grad,
                                             params.clip_norm) for grad in grads]
        train_op = optimizer.apply_gradients([*zip(grads,varis)],
                                             global_step = tf.train.get_global_step(),
                                            )
        return tf.estimator.EstimatorSpec(mode = mode,
                                          loss = loss,
                                          train_op = train_op,
                                          )   
    eval_metric_ops = {"accuracy": acc}                                    
    return tf.estimator.EstimatorSpec(mode = mode,
                                      loss = loss,
                                      eval_metric_ops = eval_metric_ops,
                                      )
    
    
class cake_train_eval(object): 
    """Train and evaluate class for model function.
    
    For training a model and/or evaluating fitness of a model on labeled 
    data. params class can be updated if needed while between training rounds. 
    
    Attributes:
        train_data: dict or npz file of dict in the form of {"x":,"y":}
        eval_data: dict or npz file of dict in the form of {"x":,"y":}
        params: params class object of model parameters for logging, training, 
                etc. See class function params. 
        model_dir: directory where model checkpoints are saved/restored
        model_fn: the model function to be fid in to tf.estimator.Estimator
    """
    
    
    def __init__(self,
                 train_data=None,
                 eval_data=None,
                 params=None,
                 model_dir=None,
                 model_fn=cake_fn,
                 ):
        """Initialization of attributes."""
        try: 
            self.train_data = np.load(train_data)
        except FileNotFoundError:
            self.train_data = train_data
        try: 
            self.eval_data = np.load(eval_data)
        except FileNotFoundError:
            self.eval_data = eval_data
        
        self.model_fn = model_fn
        self.model_dir = model_dir
        self._params = params
        self._config_set()
        self._make_model()
    
    @property
    def params(self):
        """params property fn."""
        return self._params
    
    @params.setter
    def params(self,new):
        """Sets params and updates configs and model. 
        
        Args: 
            new: a params class object. 
        """
        self._params = new
        self._config_set()
        self._make_model()
        
    def _config_set(self):
        """Makes the config for model Estimator. 
        """
        p = self._params
        self._config = tf.estimator.RunConfig(save_checkpoints_steps = p.save_checkpoints_steps,
                                              keep_checkpoint_max = p.keep_checkpoint_max,
                                              save_summary_steps = p.save_summary_steps
                                              )
        
    def _make_model(self):
        """Makes the Estimator model with model_fn. 
        """
        self._model = tf.estimator.Estimator(model_fn=self.model_fn,
                                             model_dir=self.model_dir,
                                             config=self._config,
                                             params=self._params,
                                            )   
    
    def train(self):
        """Train and evalutate method. 
        
        Returns: The results after trained model is evaluated on eval_data.
        """
        p = self._params
        if self.train_data != None:
            tens_to_log = self.params.tens_to_log
            logging_hook = tf.train.LoggingTensorHook(tensors = tens_to_log,
                                                      every_n_iter = p.logging_step,
                                                      )
            t_fn = tf.estimator.inputs.numpy_input_fn(x = {"x": self.train_data["x"]},
                                                      y = self.train_data["y"],
                                                      batch_size = p.batch_size,
                                                      num_epochs = None,
                                                      shuffle = True,
                                                      )
            self._model.train(input_fn = t_fn,
                              steps = self.params.training_steps,
                              hooks = [logging_hook],
                              )
                
        if self.eval_data != None:
            e_fn = tf.estimator.inputs.numpy_input_fn(x = {"x": self.eval_data["x"]},
                                                      y = self.eval_data["y"],
                                                      num_epochs = 1,
                                                      shuffle = False,
                                                      )
            eval_results = self.model.evaluate(input_fn = e_fn,
                                               checkpoint_path = self.model_dir,
                                               )
            print(eval_results)
            
    
class cake_predict(object): 
    """Predictor creator and evaluator for streaming input.
    
    A predictor class that is mostly generic except for the _serving_input_fn
    that specifies the serving_input_receiver_fn for tf.estimator.export. 
    Initialization with saved_path = None creates a new saved_model, and
    creates a predictor function from either the most recent saved 
    model in model_dir or a particular saved_model in 'saved_model' dir.
    Can also update the predictor with a newly trained model in model_dir. 
    Predictions are saved in a npz file as a dict with the same input dict keys 
    modified by the suffix "_pred". 
    
    Attributes:
        model_fn: (Optional) a tf model that can be fed into 
                  tf.estimator.Estimator, default is cake_fn.
        model_dir: (Optional) diretory path for the trained model, default is
                    "model".
        saved_path: (Optional) directory path, "most_recent", None (default). 
                    The "most_recent" option finds the last saved_model. By
                    default, a new saved model is created from the last model 
                    in the model_dir. 
    """
    
    def __init__(self,
                 model_fn=cake_fn,
                 model_dir: Optional[str] = "model",
                 saved_path : Optional[str] = None,
                ):
        """
        Initialization function for class variables. 
        """
        self.model_fn = model_fn   
        self.model_dir = model_dir
        if saved_path == None:
            self.update_predictor()
        elif saved_path == "most_recent":
            subdirs = [x for x in Path('saved_model').iterdir() if x.is_dir()\
                       and 'temp' not in str(x)]
            self.saved_path = "saved_model/"+str(sorted(subdirs)[-1])
            self._build_predictor()
        else:
            self.saved_path = saved_path
            self._build_predictor()
        
    def _serving_input_fn(self): 
        """Serving input_fn that builds features from placeholders
    
        Returns:
            tf.estimator.export.ServingInputReceiver
        """
        seq = tf.placeholder(dtype=tf.float32, shape=[None, None], name='seq')
        features = {'seq': seq}
        return tf.estimator.export.build_raw_serving_input_receiver_fn(features)
    
    def update_predictor(self):
        """Update predictor with newly trained model.
        
        """
        estimator = tf.estimator.Estimator(self.model_fn,
                                           self.model_dir,
                                           params={},
                                           )
        self.saved_path = estimator.export_saved_model('saved_model', 
                                                          self._serv_input_fn(),
                                                          )
        self._build_predictor()
        
    def _build_predictor(self):
        """Sets _predict_fn as a tf.contib.predictor.from_saved_model().
        
        Raises:
            OSError: When self.saved_path can't be found.
        """
        try: 
            predict_fn = tf.contrib.predictor.from_saved_model(self.saved_path)
        except OSError as err: 
            print(f"OSError: {err}")
        self._predict_fn = predict_fn
        
    def predict(self,
                data : Optional[Dict] = None,
                data_path : Optional[str] = None,
                predicted_data_dir: Optional[str] = "predictions",
                ):
        """Predict method that saves and outputs predictions.
        
        Args: 
            data: Optional dict of numpy arrays.
            data_path: Optional file path to npz file. 
            predicted_data_dir: Optional directory where predictions will be 
                                stored. 
        Returns: dict of output predictions.
        Raises:
            FileNotFoundError: If data_path isn't found.
            ValueError: If both data and data_path are specified.
        """
        if data_path != None and data == None:
            try:
                data = np.load(data_path)
            except FileNotFoundError as err:
                print(f"FileNotFoundError: {err}")
            
        elif data_path == None and data != None:
            pass
        else: 
            raise ValueError("Can not specify both 'data' and 'data_path")
        predictions={}
        for k, d in data.items():  
            predictions[k+"_pred"] = self._predict_fn({'seq': d})['output']
        keys = sorted(data.keys())
        m, M = keys[0], keys[-1]
        save_file = predicted_data_dir+f"/{m}_{M}_predicted_values"
        np.savez_compressed(save_file,**predictions)
        return predictions