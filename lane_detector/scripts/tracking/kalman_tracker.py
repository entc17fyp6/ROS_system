import numpy as np
from filterpy.kalman import KalmanFilter

class KalmanLaneTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0
    def __init__(self, polar, img=None):
        """
        Initialises a tracker using initial bounding box.
        """
        # Define constant velocity model
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        self.kf.F = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]])
        self.kf.H = np.array([[1,0,0,0],[0,1,0,0]])

        self.kf.R *= 100.
        self.kf.P *= 1000. 
        self.kf.Q *= 0.1

        self.kf.x[:2] = polar.reshape((2, 1))
        self.time_since_update = 0
        self.id = KalmanLaneTracker.count
        KalmanLaneTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, polar, img=None):
        """
        Updates the state vector with observed bbox.
        """
        print('updated')
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1

        if polar != []:
            self.kf.update(polar.reshape((2, 1)))

    def predict(self,img=None):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
#         if (self.kf.x[6] + self.kf.x[2]) <= 0:
#             self.kf.x[6] *= 0.0

        self.kf.predict()
        self.age += 1

        if self.time_since_update > 0:
            self.hit_streak = 0

        self.time_since_update += 1
        self.history.append(self.kf.x)

        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return self.kf.x