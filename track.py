class TrackStatus:
    Active = 0
    Removed = -1

class Track:
    def __init__(self, track_id, features, bbox):
        self.track_id = track_id
        self.features = features
        self.last_bbox = bbox
        self.status = TrackStatus.Active
        self.age = 1
        self.is_dirty = False
        
    def update(self, features, bbox):
        self.last_bbox = bbox
        self.features = features
        self.is_dirty = True

    def update_status(self, status):
        self.status = status