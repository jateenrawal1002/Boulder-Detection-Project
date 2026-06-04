import math
from collections import OrderedDict     #Create the dict of the ID's in order to keep them in same order for every itearation.

class EuclideanDistTracker:
    def __init__(self, max_distance_px=40, max_age=10): 
        #To set define a function that sets the initial values of the obeject to be tracked.
        #The max_distance is used to check the max distance a ID has travelled.
        #The max_age define the max miss the algorithm can do before it removes the ID.
        self.next_id = 0
        self.tracks = OrderedDict() #save the tracker ID for every single object which is currently tracked.
        self.max_distance = float(max_distance_px)
        self.max_age = int(max_age)

    @staticmethod     #Call directly from the class, no object needed.To call the function that is independent of the initial values.
    def _centroid(rect):
        x, y, w, h = rect
        return (x + w / 2.0, y + h / 2.0)
    #The tracker compares the centroid values of the consecutive frames, if the conditions of max_dist and max_age are fulfilled, it goes next to update the position of the ID.
    def reset(self):
        self.next_id = 0
        self.tracks.clear()

    def update(self, detections):
        track_ids = list(self.tracks.keys())    # make a simple list of all tracked ID
        unmatched_tracks = set(track_ids)
        unmatched_dets = set(range(len(detections)))
        pairs = []
        for tid in track_ids:
            tcx = self.tracks[tid]["cx"]    #reads the stored values, from the global dict of self.tracks-specific id and centroid of that ID
            tcy = self.tracks[tid]["cy"]
            for di, rect in enumerate(detections):
                dcx, dcy = self._centroid(rect)
                dist = math.hypot(dcx - tcx, dcy - tcy)
                if dist <= self.max_distance:
                    pairs.append((dist, tid, di))
        pairs.sort(key=lambda t: t[0])  #sort to get the best result sof the nearest detections.
        assignments, used_tracks, used_dets = [], set(), set()
        for dist, tid, di in pairs:
            if tid in used_tracks or di in used_dets:
                continue
            # The di mean the detection ID, which are the closest available Indexs for the tracking.
            assignments.append((tid, di))
            used_tracks.add(tid)
            used_dets.add(di)
            unmatched_tracks.discard(tid)
            unmatched_dets.discard(di)
        # The algorithm performs the check in the assignment list, lets the missed tracks with increment in the age 1. 
        for tid, di in assignments:
            x, y, w, h = detections[di]
            cx, cy = self._centroid(detections[di])
            self.tracks[tid].update({"cx": cx, "cy": cy, "w": w, "h": h, "missed": 0})
            self.tracks[tid]["age"] += 1
        new_ids_for_det = {}
        for di in unmatched_dets:
            x, y, w, h = detections[di]
            cx, cy = self._centroid(detections[di])
            tid = self.next_id
            self.tracks[tid] = {"cx": cx, "cy": cy, "w": w, "h": h, "age": 1, "missed": 0}
            new_ids_for_det[di] = tid
            self.next_id += 1

        dead = []
        for tid in list(self.tracks.keys()):
            if tid in unmatched_tracks:
                self.tracks[tid]["missed"] += 1
            if self.tracks[tid]["missed"] > self.max_age:
                dead.append(tid)
        for tid in dead:
            del self.tracks[tid]
        out = []
        for tid, di in assignments:
            x, y, w, h = detections[di]
            out.append([x, y, w, h, tid])
        for di in unmatched_dets:
            x, y, w, h = detections[di]
            out.append([x, y, w, h, new_ids_for_det[di]])
        return out


