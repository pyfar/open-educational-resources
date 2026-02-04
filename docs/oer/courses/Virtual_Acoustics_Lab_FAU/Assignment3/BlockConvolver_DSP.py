import numpy as np
from scipy.signal import fftconvolve

class BlockConvolver_DSP:
    """
    Block-based convolution for multiple input channels
    """
    
    def __init__(self, blockSize, numInputs, irLength):
        """
        Constructor
        
        Args:
            blockSize: Size of processing blocks
            numInputs: Number of input channels
            irLength: Length of impulse responses
        """
        self.blockSize = blockSize
        self.numInputs = numInputs
        self.irLength = irLength
        
        # Initialize impulse responses
        self.irs = np.zeros((irLength, numInputs))
        
        # Initialize overlap buffer
        self.overlap = np.zeros((irLength - 1, numInputs))
    
    def setIRs(self, irs):
        """
        Set impulse responses
        
        Args:
            irs: Impulse responses [irLength x numInputs]
        """
        assert irs.shape[0] == self.irLength, "IR length must match"
        assert irs.shape[1] == self.numInputs, "Number of IRs must match number of inputs"
        self.irs = irs
    
    def process(self, inSig):
        """
        Process input signal through convolver
        
        Args:
            inSig: Input signal matrix [blockSize x numInputs]
            
        Returns:
            np.ndarray: Output signal matrix [blockSize x numInputs]
        """
        assert inSig.shape[0] == self.blockSize, "Input block size must match"
        assert inSig.shape[1] == self.numInputs, "Number of inputs must match"
        
        # Perform convolution
        outSig = np.zeros((self.blockSize + self.irLength - 1, self.numInputs))
        for i in range(self.numInputs):
            outSig[:, i] = fftconvolve(inSig[:, i], self.irs[:, i])
        
        # Add overlap from previous block
        outSig[:self.irLength-1] += self.overlap
        
        # Store overlap for next block
        self.overlap = outSig[self.blockSize:]
        
        # Return only the current block
        return outSig[:self.blockSize] 