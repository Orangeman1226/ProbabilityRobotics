class OccSensorNoisemark:
    def __init__(self):
        self.pos = []
        self.id = None
        self.Noise_code = ""
        self.Hasdrawn = False 

    def draw(self,ax,elems):

        if len(self.Noise_code) >= 1 :       
            c = ax.scatter(self.pos[0], self.pos[1], s=100, marker="+",label="Noisemarks",color="yellow")
            elems.append(c)
            elems.append(ax.text(-6.25,-8.5,"************\n"+"!!SensorNoise_"+self.Noise_code+"!!\n************",fontsize=10,color="red"))
            self.Hasdrawn = True
    
        self.Noise_code = ""
    
    
