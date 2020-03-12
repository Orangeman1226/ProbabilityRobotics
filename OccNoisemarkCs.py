class OccNoisemark:
    def __init__(self):
        self.pos = []
        self.id = None
        self.Noise_code = ""
        self.Hasdrawn = False 

    def draw(self,ax,elems):

        if len(self.Noise_code) == 3 and self.Noise_code != "---" :       
            c = ax.scatter(self.pos[0], self.pos[1], s=100, marker="^",label="Noisemarks",color="blue")
            elems.append(c)
            elems.append(ax.text(6.25,-8.5,"************\n"+"!!Noise_"+self.Noise_code+"!!\n************",fontsize=10))
            self.Hasdrawn = True
    
        self.Noise_code = ""
    
    
