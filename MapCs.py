class Map:
    def __init__(self):
        self.landmarks = []
        self.noisemarks = []
    
    def append_landmark(self,landmark):
        landmark.id = len(self.landmarks)
        self.landmarks.append(landmark)

    def draw(self ,ax, elems):
        for lm in self.landmarks:lm.draw(ax,elems)

