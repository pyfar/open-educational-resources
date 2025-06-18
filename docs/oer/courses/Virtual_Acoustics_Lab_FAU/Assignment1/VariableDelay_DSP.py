import numpy as np
from scipy.interpolate import interp1d
from DSP import DSP
from Config import Config
import matplotlib.pyplot as plt

class VariableDelay_DSP(DSP):
    """
    Multiple input, multiple output delay line.
    """
    
    def __init__(self, numberOfInputs, config):
        """Initialize the Variable Delay DSP."""
        self.blockSize = config.blockSize
        self.numberOfInputs = numberOfInputs
        self.numberOfOutputs = numberOfInputs  # Output is delayed input
        self.maximumDelay = config.maximumDelay
        self.lastDelay = np.zeros((self.numberOfInputs, self.numberOfOutputs))
        self.newDelay = self.lastDelay
        self.iter = 0
        self.maxSize = 100
        self.blockStack = np.zeros((config.blockSize * self.maxSize, numberOfInputs))
        self.checkConfig()
        
    
    def process(self, inSig, prevBlock=np.zeros((1024,1))):
        """Process the input signal with time-varying delays."""
        
        blockSize = inSig.shape[0]
        delayCurve = self.interpolateDelays(blockSize)
        outSig = np.zeros_like(inSig)
        
        maxSize = self.maxSize
        self.blockStack[np.arange(blockSize * (maxSize - 1)),:] = self.blockStack[np.arange(blockSize, blockSize * maxSize),:]
        self.blockStack[np.arange(blockSize* (maxSize - 1), blockSize * maxSize),:] = inSig
        
        for ch in range(self.numberOfInputs):
            interpFunc = interp1d(np.arange(blockSize * maxSize), self.blockStack[:, ch], kind='linear', fill_value="extrapolate")
            delayedIndices = np.arange(blockSize) - delayCurve[:, ch] + blockSize * (maxSize - 1)

            outSig[:, ch] = interpFunc(delayedIndices)
        
        self.lastDelay = self.newDelay
        return outSig
        
    
    def setDelay(self, newDelay):
        """Set new delay values."""
        assert newDelay.shape[1] == self.numberOfInputs, "Incorrect size of delays"
        self.newDelay = newDelay     
        
    
    def interpolateDelays(self, blockSize):
        """Linear interpolation between delay values on a block-by-block basis."""
        linearFadeIn = np.linspace(0, 1, blockSize).reshape(-1, 1)
        linearFadeOut = np.linspace(1, 0, blockSize).reshape(-1, 1)
        fadedDelay = linearFadeIn * self.newDelay + linearFadeOut * self.lastDelay
        return fadedDelay.reshape(blockSize, self.numberOfInputs)

if __name__ == "__main__":

    # Define Config = Fixed during runtime
    config = Config()

    # Setup DSP object
    numberOfInputs = 1
    delay = VariableDelay_DSP(numberOfInputs, config)

    # Create input signal
    numberOfBlocks = 10
    length = config.blockSize * numberOfBlocks
    rampSignal = np.linspace(0, 1, length).reshape(-1,1)

    # Set target delays
    targetDelays = np.random.rand(numberOfBlocks, numberOfInputs) * 200

    outputSignal = np.zeros((length, numberOfInputs))

    prevBlock = np.zeros((config.blockSize, numberOfInputs))
    for it in range(numberOfBlocks):
        delay.setDelay(targetDelays[it,:].reshape(numberOfInputs, 1))
        blockIndex = np.arange(config.blockSize) + it * config.blockSize
        block = rampSignal[blockIndex]
        outputSignal[blockIndex, :] = delay.process(block)
        prevBlock = block

    # Plot
    plt.figure()
    plt.grid(True)
    plt.plot(rampSignal, label='Input')
    plt.plot(outputSignal, label='Output')
    plt.legend()
    plt.xlabel('Time [samples]')
    plt.ylabel('Amplitude [lin]')
    plt.show()
