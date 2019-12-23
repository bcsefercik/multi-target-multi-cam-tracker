class Tracklet:
    def __init__(self, tracklet_id, max_length, bbox):
        self.tracklet_id = tracklet_id
        self.predicted_ids = {}
        self.max_length = max_length
        self.last_bbox = bbox
        self.age = 0

    def update(self, new_id):
        if not new_id in self.predicted_ids:
            self.predicted_ids[new_id] = 1
        else:
            self.predicted_ids[new_id] += 1 

    def update_bbox(self, new_bbox):
        # print(new_bbox)
        self.last_bbox = new_bbox

    def get_current_length(self):
        return sum(self.predicted_ids.values())

    def is_done(self):
        return self.get_current_length() == self.max_length

    def get_purity(self):
        return max(self.predicted_ids.values())/self.max_length

class TrackletManager():
    def __init__(self, tracklet_length, tracker, file_path):
        self.tracklet_length = tracklet_length
        self.tracker = tracker
        self.tracklets = {}
        self.total_purity = 0.0
        self.total_purity_count = 0
        self.next_tracklet_id = 1
        self.g_count = 0
        self.file_path = file_path
        with open(self.file_path, "w") as f:
            f.write("ID\tPurity\n")
        print("ID\tPurity\t")

    def update(self, gt_detections):
        # print("-----------")
        # TODO: increase age for all tracklets
        r = False
        tracklet_ids = list(self.tracklets.keys())
        bbbb = []
        for ID in gt_detections:
            if not ID in self.tracklets:
                self.tracklets[ID] = Tracklet(self.next_tracklet_id, self.tracklet_length, list(gt_detections[ID]["bbox"]))
                self.next_tracklet_id += 1
                tracklet_ids.append(ID)
            else:
                self.tracklets[ID].update_bbox(list(gt_detections[ID]["bbox"]))

            for ti, track in enumerate(self.tracker.tracks):
                if not list(track.last_bbox) in bbbb:
                    bbbb.append(list(track.last_bbox))
                    if self.tracklets[ID].last_bbox == list(track.last_bbox):
                        self.tracklets[ID].update(track.track_id)
                        tracklet_ids.remove(ID)
                    # print(self.tracklets[ID].predicted_ids)

        for i in tracklet_ids:
            self.tracklets[i].update(-1)
            self.tracklets[i].last_bbox = None

        
        keys_to_pop = set()
        for tkey in self.tracklets:
            if self.tracklets[tkey].is_done():
                self.total_purity += self.tracklets[tkey].get_purity()
                self.total_purity_count += 1
                # print(self.total_purity/self.total_purity_count)
                if self.tracklets[tkey].get_purity() > 0.999:
                    self.g_count += 1
                log_text = "{}\t{}\n".format(self.tracklets[tkey].tracklet_id, self.tracklets[tkey].get_purity())
                with open(self.file_path, "a") as f:
                    f.write(log_text)
                # print()
                keys_to_pop.add(tkey)

        for p in keys_to_pop:
            self.tracklets.pop(p)
        
        return r
