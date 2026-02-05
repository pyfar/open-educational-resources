import numpy as np

class DSP:
    """
    API for DSP units.
    Provided
    """
    
    def __init__(self):
        self.numberOfInputs = 0
        self.numberOfOutputs = 0
        self.blockSize = 256
    
    def checkConfig(self):
        assert self.numberOfInputs > 0, "Set Input number"
        assert self.numberOfOutputs > 0, "Set Output number"
        output = self.process(np.zeros((self.blockSize, self.numberOfInputs)))
       # assert output.shape[1] == self.numberOfOutputs, "Output size mismatch"
    