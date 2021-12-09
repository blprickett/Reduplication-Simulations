import tensorflow as tf
import numpy as np
import time

class Encoder(tf.keras.Model):
  def __init__(self, enc_units, batch_sz, layer_type, dropout):
    super(Encoder, self).__init__()
    self.batch_sz = batch_sz
    self.enc_units = enc_units
    self.layer_type = layer_type
    self.dropout = dropout
    
    if layer_type == "gru":
        self.layer1 = tf.keras.layers.GRU(self.enc_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform',
                                       name="enc_1",
                                       dropout=self.dropout, 
                                       recurrent_dropout=self.dropout)
        self.layer2 = tf.keras.layers.GRU(self.enc_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform',
                                       name="enc_2",
                                       dropout=self.dropout, 
                                       recurrent_dropout=self.dropout)
    elif layer_type == "lstm":
        self.layer1 = tf.keras.layers.LSTM(self.enc_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform',
                                       name="enc_1",
                                       dropout=self.dropout, 
                                       recurrent_dropout=self.dropout)
        self.layer2 = tf.keras.layers.LSTM(self.enc_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform',
                                       name="enc_2",
                                       dropout=self.dropout, 
                                       recurrent_dropout=self.dropout)
    else:
        raise Exception("Layer type must be 'lstm' or 'gru'!")

  def call(self, x, hidden):
    if self.layer_type == "gru":
        output1, state1 = self.layer1(x, initial_state = hidden)
        output2, state2 = self.layer2(output1, initial_state = hidden)
        return output2, state1, state2
    elif self.layer_type == "lstm":
        output1, state1a, state1b = self.layer1(x, initial_state = [hidden, hidden])
        output2, state2a, state2b = self.layer2(output1, initial_state = [hidden, hidden])
        return output2, state1a, state1b, state2a, state2b    
    else:
        raise Exception("Layer type must be 'lstm' or 'gru'!")
        
  def initialize_hidden_state(self, this_bs):
    return tf.zeros((this_bs, self.enc_units))
	
class Decoder(tf.keras.Model):
  def __init__(self, dec_units, batch_sz, output_dim, layer_type, dropout):
    super(Decoder, self).__init__()
    self.batch_sz = batch_sz
    self.dec_units = dec_units
    self.layer_type = layer_type
    self.dropout = dropout

    if layer_type == "gru":
        self.layer1 = tf.keras.layers.GRU(self.dec_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform',
                                       name="dec_1",
                                       dropout=self.dropout, 
                                       recurrent_dropout=self.dropout)
        self.layer2 = tf.keras.layers.GRU(self.dec_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform',
                                       name="dec_2",
                                       dropout=self.dropout, 
                                       recurrent_dropout=self.dropout)
    elif layer_type == "lstm":
        self.layer1 = tf.keras.layers.LSTM(self.dec_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform',
                                       name="dec_1",
                                       dropout=self.dropout, 
                                       recurrent_dropout=self.dropout)
        self.layer2 = tf.keras.layers.LSTM(self.dec_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform',
                                       name="dec_2",
                                       dropout=self.dropout, 
                                       recurrent_dropout=self.dropout)
    else:
        raise Exception("Layer type must be 'lstm' or 'gru'!")
        
    self.output_layer = tf.keras.layers.Dense(output_dim, activation="tanh")

  def call(self, x, enc_state1a, enc_state2a, enc_output, enc_state1b=None, enc_state2b=None):
    if self.layer_type == "lstm":
        x, state1a, state1b = self.layer1(x, initial_state=[enc_state1a, enc_state1b])
        x, state2a, state2b = self.layer2(x, initial_state=[enc_state2a, enc_state2b])
        
        # output shape == (batch_size * 1, hidden_size)
        x = tf.reshape(x, (-1, x.shape[2]))

        # output shape == (batch_size, feature num)
        output = self.output_layer(x)

        return output, state1a, state1b, state2a, state2b
    elif self.layer_type == "gru":
        x, state1 = self.layer1(x, initial_state=enc_state1a)
        x, state2 = self.layer2(x, initial_state=enc_state2a)
        
        # output shape == (batch_size * 1, hidden_size)
        x = tf.reshape(x, (-1, x.shape[2]))

        # output shape == (batch_size, feature num)
        output = self.output_layer(x)

        return output, state1, state2 
	
class seq2seq ():
  def __init__(self, hidden_dim, batch_size, output_length, input_dim, output_dim, learn_rate, layer_type, dropout=0.0):
    self.encoder = Encoder(hidden_dim, batch_size, layer_type, dropout)
    self.decoder = Decoder(hidden_dim, batch_size, output_dim, layer_type, dropout)
    self.optimizer, self.loss_object = self.init_optimizer(learn_rate)
    self.units = hidden_dim
    self.output_length = output_length
    self.output_dim = output_dim
    self.input_dim = input_dim
    self.bs = batch_size
    self.layer_type = layer_type
    self.droupout = dropout
   
  def init_optimizer (self, lr):
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    loss = tf.keras.losses.MeanSquaredError()
    return opt, loss
  
  def loss_function(self, real, pred):
    loss_ = self.loss_object(real, pred)
    return tf.reduce_mean(loss_)
      
  @tf.function
  def train_step(self, inp, full_targ, this_bs):
    enc_hidden = self.encoder.initialize_hidden_state(this_bs)
    loss = 0
    with tf.GradientTape() as tape:
      if self.layer_type == "gru":
        enc_output, enc_hidden1, enc_hidden2 = self.encoder(inp, enc_hidden)
        dec_input = tf.expand_dims([[0.0 for u in range(self.output_dim)]] * this_bs, 1)
        dec_hidden1, dec_hidden2 = enc_hidden1, enc_hidden2
      elif self.layer_type == "lstm":
        enc_output, enc_hidden1a, enc_hidden1b, enc_hidden2a, enc_hidden2b = self.encoder(inp, enc_hidden)
        dec_input = tf.expand_dims([[0.0 for u in range(self.output_dim)]] * this_bs, 1)
        dec_hidden1a, dec_hidden1b, dec_hidden2a, dec_hidden2b = enc_hidden1a, enc_hidden1b, enc_hidden2a, enc_hidden2b

      # Teacher forcing - feeding the target as the next input
      for t in range(full_targ.shape[1]):
        if self.layer_type == "lstm":
            dec_output, dec_hidden1a, dec_hidden1b, dec_hidden2a, dec_hidden2b = self.decoder(dec_input, dec_hidden1a, dec_hidden2a, enc_output, dec_hidden1b, dec_hidden2b)
        elif self.layer_type == "gru":   
            dec_output, dec_hidden1, dec_hidden2 = self.decoder(dec_input, dec_hidden1, dec_hidden2, enc_output)
        loss += self.loss_function(full_targ[:, t], dec_output)

        # using teacher forcing
        dec_input = tf.expand_dims(full_targ[:, t], 1)

    variables = self.encoder.trainable_variables + self.decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    self.optimizer.apply_gradients(zip(gradients, variables))

    return loss
      
  def train (self, X, Y, epoch_num, print_every):
    raw_data = (X, Y)
    buffer_size = X.shape[0]
    dataset = tf.data.Dataset.from_tensor_slices(raw_data).shuffle(buffer_size)
    my_data = dataset.batch(self.bs, drop_remainder=True)
    steps_per_epoch = X.shape[0]//self.bs
    
    learning_curve = {"Loss":[]}
      
    start = time.time()  
    for epoch in range(epoch_num):
      total_loss = 0
      for (batch, X_Y) in enumerate(my_data.take(steps_per_epoch)):
        inp = X_Y[0]
        full_target = X_Y[1]
        this_loss = self.train_step(inp, full_target, self.bs)
        total_loss += this_loss
        
      if (epoch+1) % print_every == 0:
        print('Epoch {} Loss {:.8f}'.format(epoch + 1,
                                          total_loss / steps_per_epoch))
        print('Time taken for {} epochs {} sec\n'.format(print_every, time.time() - start))
        start = time.time()
        
      #Record learning
      learning_curve["Loss"].append(total_loss.numpy() / steps_per_epoch)          

    return learning_curve
    
  def single_step (self, X, Y):
    raw_data = (X, Y)
    buffer_size = X.shape[0]
    dataset = tf.data.Dataset.from_tensor_slices(raw_data).shuffle(buffer_size)
    my_data = dataset.batch(1, drop_remainder=True)
    steps_per_epoch = X.shape[0]//1
    #print(steps_per_epoch, X.shape)
    for (batch, X_Y) in enumerate(my_data.take(steps_per_epoch)):
        inp = X_Y[0]
        full_target = X_Y[1]
        this_loss = self.train_step(inp, full_target, 1)
        break
     

    return this_loss
      
  def predict(self, X):
    Y_hat = []
    for x in X:
      if self.layer_type == "gru":
        hidden = self.encoder.initialize_hidden_state(self.bs)
        enc_out, enc_hidden1, enc_hidden2 = self.encoder(np.array([x]), hidden)
        dec_input = tf.expand_dims([[0.0 for u in range(self.output_dim)]] * self.bs, 1)
        dec_hidden1, dec_hidden2 = enc_hidden1, enc_hidden2
      elif self.layer_type == "lstm":
        hidden = self.encoder.initialize_hidden_state(self.bs)
        enc_out, enc_hidden1a, enc_hidden1b, enc_hidden2a, enc_hidden2b = self.encoder(np.array([x]), hidden)
        dec_input = tf.expand_dims([[0.0 for u in range(self.output_dim)]] * self.bs, 1)
        dec_hidden1a, dec_hidden1b, dec_hidden2a, dec_hidden2b = enc_hidden1a, enc_hidden1b, enc_hidden2a, enc_hidden2b
        
      x_results = []
      for t in range(self.output_length):
          #Run the decoder for this timestep:
          if self.layer_type == "gru":
            dec_output, dec_hidden1, dec_hidden2 = self.decoder(dec_input, dec_hidden1, dec_hidden2, enc_out)
          elif self.layer_type == "lstm":
            dec_output, dec_hidden1a, dec_hidden1b, dec_hidden2a, dec_hidden2b = self.decoder(dec_input, dec_hidden1a, dec_hidden1b, enc_out, dec_hidden2a, dec_hidden2b)

          #Save the output:
          x_results.append(dec_output[0])

          #The output is then fed back into the model:
          dec_input = tf.expand_dims(dec_output, 1)
      Y_hat.append(np.array(x_results))

    return np.array(Y_hat)
    
