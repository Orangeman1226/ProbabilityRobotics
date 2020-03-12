import matplotlib
matplotlib.use('nbagg')
import matplotlib.animation as anm

import matplotlib.pyplot as plt
from tqdm import tqdm

tqdm_pbar = tqdm()#処理進捗のインジケータ
      
class World:
    @classmethod
    def getWorldRange(cls):
        range_maxLim = 10
        range_minLim = -10
        return (range_minLim,range_maxLim)
    

    #time_span     : the span for the simuratiuon
    #time_interval : delta_t for calculating
    def __init__(self,time_span,time_interval,debug = False):
      self.objects =[]
      self.debug = debug
      self.time_span = time_span
      self.time_interval = time_interval
      self.word_range = self.getWorldRange()
            
    
    def append(self,obj):
     self.objects.append(obj)


    def draw(self):
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')
        
        ax.set_xlim(self.word_range[0],self.word_range[1])
        ax.set_ylim(self.word_range[0],self.word_range[1])
        
        ax.set_xlabel("X",fontsize=20)
        ax.set_ylabel("Y",fontsize=20)

        elems = []
         
        
        if self.debug:

            for i in range(1000):self.one_step(i,elems,ax)

        else:
            
            #処理進捗のインジケータ
            tqdm_pbar.total = int(self.time_span/self.time_interval)+2

            self.ani = anm.FuncAnimation(fig,self.one_step,fargs=(elems,ax)
                        ,frames=int(self.time_span/self.time_interval)+1,interval=int(self.time_interval*1000),repeat=False,blit=True)
            #plt.show()
            self.ani.save("Simulatting_Robot.gif", writer = 'pillow')
            print("******Completed to save Anim gif !!******")
            tqdm_pbar.close()#処理進捗のインジケータ終了
            
    def  one_step(self,i,elems,ax):
        while elems:
             elems.pop().remove()
        
        elems.append(ax.text(-7.5,7.5,"t="+ str(i),fontsize=15))

        for obj in self.objects: 
            obj.draw(ax,elems)
            if hasattr(obj,"one_step"):
                obj.one_step(1.0)
        
        tqdm_pbar.update(1)#処理進捗のインジケータ更新

         