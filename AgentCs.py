class Agent:
    def __init__(self,nu,omega):
     self.nu = nu
     self.omega = omega
    
    def decision(self,observation=None):
         return self.nu,self.omega
    