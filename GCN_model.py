# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 12:26:18 2021

@author: L.Badr
"""
from spektral.layers import GCNConv

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from spektral.utils.convolution import gcn_filter
from sklearn.preprocessing import MinMaxScaler

class GCN_model:
    def __init__(self, X, y, A, epochs = 200):
        self.X = X
        self.y = y
        self.A = A
        self.F = X.shape[1] #the size of node features
        self.N = X.shape[0] #the number of nodes
    
    def build(self):
        dropout = 0.5           # Dropout rate for the features
        X_in = Input(shape=(self.F, ))
        fltr_in = Input((self.N, ), sparse=True)        
        
        graph_conv_1 = GCNConv(512,kernel_initializer='normal',
                                 activation='relu')([X_in, fltr_in])
        
        dropout_2 = Dropout(dropout)(graph_conv_1)
        graph_conv_2 = GCNConv(256,
                                 activation='relu')([dropout_2, fltr_in])
        
        dropout_3 = Dropout(dropout)(graph_conv_2)
        graph_conv_3 = GCNConv(128,
                                 activation='relu')([dropout_3, fltr_in])
        
        dropout_4 = Dropout(dropout)(graph_conv_3)
        graph_conv_4 = GCNConv(1)([dropout_4, fltr_in])
        
        # Build model
        self.model = Model(inputs=[X_in, fltr_in], outputs=graph_conv_4)
        self.model.compile(optimizer='adam',
                      loss='mean_absolute_error',
                      weighted_metrics=['mae','mse'])
        self.model.summary()
    
    def fit(self, train_mask, valid_mask, n_epochs = 200):   
    
        self.A_process= gcn_filter(self.A) #preprocess the adjacency matrix
    
        self.X_scaled = MinMaxScaler().fit_transform(self.X) #Normalizing X data
    
        validation_data = ([self.X_scaled, self.A_process], self.y, valid_mask) #validation set
        
        self.model.fit([self.X_scaled, self.A_process],
          self.y,
          sample_weight=train_mask,
          epochs=n_epochs,
          batch_size=self.N, #whole dataset as batch to avoid shuffle
          validation_data=validation_data,
          shuffle=False,
          callbacks=[
              EarlyStopping(patience=50,  restore_best_weights=True)
          ])
    
    def predict(self, test_mask):
        y_pred = self.model.predict([self.X_scaled , self.A_process], batch_size=self.N)
        return y_pred[test_mask]
    