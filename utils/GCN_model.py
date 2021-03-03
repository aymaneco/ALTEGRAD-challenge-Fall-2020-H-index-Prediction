from spektral.layers import GCNConv

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from spektral.utils.convolution import gcn_filter
from sklearn.preprocessing import MinMaxScaler


class GCN_model:
    def __init__(self, X, y, A, epochs=200):
        '''       
        Parameters
        ----------
        X : 2D-array float
            matrix of shape NxF, with N number of nodes and F number of features
        y : 1D-array float
            target variable 
        A : 2D-array float
            Adjacency matrix of the graph with shape NxN 
        epochs : integer, optional
            Number of epochs. The default is 200.
        '''
        self.X = X
        self.y = y
        self.A = A
        self.F = X.shape[1]  # the size of node features
        self.N = X.shape[0]  # the number of nodes

    def build(self):
        '''
        Set the architecture of the GCN model and compile it
        '''
        dropout = 0.5           # Dropout rate for the features
        X_in = Input(shape=(self.F, ))
        fltr_in = Input((self.N, ), sparse=True)

        graph_conv_1 = GCNConv(512, kernel_initializer='normal',
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
                           weighted_metrics=['mae', 'mse'])
        self.model.summary()

    def fit(self, train_mask, valid_mask, n_epochs=200):
        '''
        Fit the model, it should be run after build() function

        Parameters
        ----------
        train_mask : 1D-array of boolean of size N
            Mask that specify the train set, where True is the train observations in X.
        valid_mask : 1D-array of boolean of size N
            Mask that specify the validation set, where True is the validation observations in X.
        n_epochs : integer, optional
            number of epochs for training. The default is 200.

        '''
        self.A_process = gcn_filter(self.A)  # preprocess the adjacency matrix

        self.X_scaled = MinMaxScaler().fit_transform(self.X)  # Normalizing X data

        validation_data = ([self.X_scaled, self.A_process],
                           self.y, valid_mask)  # validation set

        self.model.fit([self.X_scaled, self.A_process],
                       self.y,
                       sample_weight=train_mask,
                       epochs=n_epochs,
                       batch_size=self.N,  # whole dataset as batch to avoid shuffle
                       validation_data=validation_data,
                       shuffle=False,
                       callbacks=[
            EarlyStopping(patience=50,  restore_best_weights=True)
        ])

    def predict(self, test_mask):
        '''
        Predict on the whole dataset then output the predictions for test observations
        that are specified with test_mask.

        Parameters
        ----------
        test_mask : 1D-array of boolean of size N
            Mask that specify the validation set, where True is the test observations in X.

        Returns
        -------
        1D-array float with size equal the number of True values in test_mask 
            Return the prediction for the test_mask subset only.
        '''
        y_pred = self.model.predict(
            [self.X_scaled, self.A_process], batch_size=self.N)
        return y_pred[test_mask]
