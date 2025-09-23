import numpy as np
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from scipy.linalg import block_diag


class Tracker:
    # Tracker for x=[x,x',y,y'], z=[x,y]
    def __init__(self, trackId=0, P=100.0,R_std=0.35,Q_std=0.04,dim_x=4, dim_z=2,dim_Q=2,block_size=2,x_TO_z=1.):
        # Initialize parameters for tracker
        self.id = trackId # assign the ID for the tracker
        self.num_matched = 0
        self.num_unmatched = 0
        self.position = []
        #lei
        self.bbx_img = []
        self.img_feat = []
        self.classes = []
        self.sf_label = 100
        self.rnn_position_list = []
        self.rnn_position = []
        self.dist_rnn = []
        #self.x_img_previous = []
        self.distance_pos=[]

        # Initialize parameters for the Kalman filter
        self.kf = KalmanFilter(dim_x=dim_x, dim_z=dim_z)
        self.dt = 1.0
        self.x_previous = []  #change to name as self.x_previous

        # State transition matrix (assuming constant velocity model)
        self.kf.F = np.array([[1, self.dt, 0,       0],
                              [0,       1, 0,       0],
                              [0,       0, 1, self.dt],
                              [0,       0, 0,       1]])

        # Measurement matrix (convert x to a measurement z)
        self.kf.H = np.array([[x_TO_z, 0, 0, 0],
                              [0, 0, x_TO_z, 0]])
        
        #KalmanFilter initializes R, P, and Q to the identity matrix, 
        #so kf.P *= P is one way to quickly assign all of the diagonal elements to the same scalar value. 
        
        # State covariance matrix
        self.kf.P *= P

        # Process uncertainty
        self.kf.Q = Q_discrete_white_noise(dim=dim_Q, dt=self.dt, var=Q_std**2, block_size=block_size)
        # State uncertainty
        #self.kf.R = np.eye(4)*6.25
        self.kf.R *= R_std**2

    # Predict and Update the next state for a position
    def predict_and_update(self, z):
        """
        :param z: centroid
        :return: centroid with updated location
        """
        self.kf.x = self.x_previous #self.kf.x

        # Predict
        self.kf.predict()

        # Update
        self.kf.update(z)
        # Get current state
        # # round to integers for pixel coordinates,then current state will be previous state
        # self.x_previous =self.kf.x.astype(float).round().astype(int)
        # keep float values,the current state will be previous state
        self.x_previous =self.kf.x.astype(float)

    # Only predict the next state for a position
    def predict(self):
        """
        :return: centroid with predicted location
        """
        self.kf.x = self.x_previous #or self.kf.x = self.position only if self.position is a form like x=[x,x',y,y']; Here is not

        # Predict
        self.kf.predict()

        # Get current state
        # # round to integers for pixel coordinates,then current state will be previous state
        # self.x_previous =self.kf.x.astype(float).round().astype(int)
        # keep float values,the current state will be previous state
        self.x_previous =self.kf.x.astype(float)


