"""
    SORT: A Simple, Online and Realtime Tracker
    Copyright (C) 2016 Alex Bewley alex@dynamicdetection.com

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

from __future__ import print_function
import numpy as np

from .kalman_tracker import KalmanLaneTracker
from .data_association import associate_detections_to_trackers


class Sort:

    def __init__(self, max_age=3, min_hits=3, use_dlib = False):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.trackers = []
        self.frame_count = 0
        self.previous = {}

        self.use_dlib = use_dlib

    def update(self, dets, img=None):
        """
        Params:
          dets - a numpy array of detections in the format [[rho,theta],[rho,theta],...]
        Requires: this method must be called once for each frame even with empty detections.
        Returns the a similar array, where the last column is the object ID.

        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        
        # for tracker in self.trackers:
            # print("time since last update ",tracker.time_since_update)
            # print('track id ',tracker.id)

        self.frame_count += 1
        # Get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 2))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict(img) #for kal!

            trk[:] = [pos[0], pos[1]]

            if (np.any(np.isnan(pos))):
                to_del.append(t)

        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))

        for t in reversed(to_del):
            self.trackers.pop(t)

        if dets.shape[0] != 0:
            matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets,trks)

            # # Update matched trackers with assigned detections
            for t, trk in enumerate(self.trackers):
                if t not in unmatched_trks:
                    d = matched[np.where(matched[:,1] == t)[0], 0]
                    trk.update(dets[d,:][0])


            # Create and initialise new trackers for unmatched detections
            for i in unmatched_dets:

                trk = KalmanLaneTracker(dets[i,:])
                self.trackers.append(trk)

        i = len(self.trackers)

        for trk in reversed(self.trackers):

            d = trk.get_state()

            cancat = np.append(d,[trk.id+1])

            ret.append(cancat.reshape(1,-1))
            i -= 1

            # Remove dead tracklet
            if trk.time_since_update> self.max_age:
                # print("popped")
                self.trackers.pop(i)

        if len(ret) > 0:
            return np.concatenate(ret)

        return np.empty((0, 6))