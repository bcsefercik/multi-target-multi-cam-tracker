class Tracklet:
    def __init__(self, tracklet_id, max_length, bbox):
        self.tracklet_id = tracklet_id
        self.predicted_ids = {}
        self.max_length = max_length
        self.last_bbox = bbox
        self.age = 0
        self.is_done_flag = False

    def update(self, new_id, is_id_old=True):
        if self.predicted_ids and new_id not in self.predicted_ids:
                self.is_done_flag = True
        
        if (not is_id_old) and self.predicted_ids:
            new_id = -1

        if new_id not in self.predicted_ids:
            self.predicted_ids[new_id] = 1
        else:
            self.predicted_ids[new_id] += 1 

    def update_bbox(self, new_bbox):
        # print(new_bbox)
        self.last_bbox = new_bbox

    def get_current_length(self):
        return max(self.predicted_ids.values())

    def is_done(self):
        return self.is_done_flag or self.get_current_length() == self.max_length

    def get_purity(self):
        return max(self.predicted_ids.values())/self.max_length if len(self.predicted_ids) > 1 and (-1 not in self.predicted_ids) else 1.0

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
        self.last_track_ids = list(map(lambda x: x.track_id, self.tracker.tracks))
        open(self.file_path, "w").close()
        # with open(self.file_path, "w") as f:
        #     f.write("gtID\tPurity\n")

    def update(self, gt_detections):
        # print("-----------")
        # TODO: increase age for all tracklets
        # print(" ".join(map(lambda x: str(x.track_id), self.tracker.tracks)))
        r = False
        tracklet_ids = list(self.tracklets.keys())
        bbbb = []

        for gtID in gt_detections:
            if not gtID in self.tracklets:
                self.tracklets[gtID] = Tracklet(self.next_tracklet_id, self.tracklet_length, list(gt_detections[gtID]["bbox"]))
                self.next_tracklet_id += 1
                # tracklet_ids.append(gtID)
            else:
                self.tracklets[gtID].update_bbox(list(gt_detections[gtID]["bbox"]))
                
            for ti, track in enumerate(self.tracker.tracks): 
                if not list(track.last_bbox) in bbbb:
                    bbbb.append(list(track.last_bbox))

                if self.is_list_equal(self.tracklets[gtID].last_bbox, list(track.last_bbox)):
                    # if gtID == 2:
                    #     print(gtID, track.track_id, self.tracklets[gtID].predicted_ids)
                    #     print(self.tracklets[gtID].get_purity(), self.tracklets[gtID].get_current_length())
                    if track.track_id in self.last_track_ids:
                        self.tracklets[gtID].update(track.track_id)
                    else:
                        self.tracklets[gtID].update(track.track_id, is_id_old=False)

                    if gtID in tracklet_ids:
                        tracklet_ids.remove(gtID)

        for i in tracklet_ids:
            self.tracklets[i].update(-1)
            self.tracklets[i].last_bbox = None

        
        keys_to_pop = set()
        for tkey in self.tracklets:
            if self.tracklets[tkey].is_done():
                self.total_purity += self.tracklets[tkey].get_purity()
                self.total_purity_count += 1

                if self.tracklets[tkey].get_purity() > 0.999:
                    self.g_count += 1
                
                log_text = f"{self.tracklets[tkey].tracklet_id}\t{self.tracklets[tkey].get_purity()}\t{self.tracklets[tkey].get_current_length()}\n"
                with open(self.file_path, "a") as f:
                    f.write(log_text)
                
                # if self.tracklets[tkey].get_purity() > 0.999 and self.tracklets[tkey].get_current_length() != self.tracklets[tkey].max_length:
                # if self.tracklets[tkey].get_purity() < 1.0:
                    # print(log_text)
                # print(self.tracklets[tkey].predicted_ids)
                keys_to_pop.add(tkey)

        for p in keys_to_pop:
            self.tracklets.pop(p)
        
        self.last_track_ids = list(map(lambda x: x.track_id, self.tracker.tracks))

        return r

    def is_list_equal(self, b1, b2):
        r = True

        for i in range(len(b1)):
            r = r and b1[i] == b2[i]

        return r

