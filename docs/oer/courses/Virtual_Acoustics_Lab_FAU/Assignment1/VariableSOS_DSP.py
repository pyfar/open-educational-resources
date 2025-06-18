import numpy as np
import pyfar as pf
import DSP
from DSP import DSP
import scipy.signal as spsignal
import matplotlib.pyplot as plt

class VariableSOS_DSP(DSP):
    """
    SISO biquad filter with time-varying behavior.
    """
    
    def __init__(self, config):
        
        self.sosMatrixLast = np.array([[1, 0, 0, 1, 0, 0]])
        self.sosMatrixNew = np.array([[1, 0, 0, 1, 0, 0]])
        
        self.biquadA = pf.FilterSOS(self.sosMatrixLast, sampling_rate=config.fs)
        self.biquadB = pf.FilterSOS(self.sosMatrixNew, sampling_rate=config.fs)

        self.biquadA.init_state((1,))
        self.biquadB.init_state((1,))    
        
        self.fadeOut = np.linspace(1, 0, config.blockSize)
        self.fadeIn = np.linspace(0, 1, config.blockSize)
        
        self.numberOfInputs = 1
        self.numberOfOutputs = 1
        self.blockSize = config.blockSize
        self.fs = config.fs
        
        self.checkConfig()
    
    def process(self, inSig):
        """Process the input signal."""
        self.biquadA.coefficients = self.sosMatrixLast
        self.biquadB.coefficients = self.sosMatrixNew
        
        inSig = np.squeeze(inSig)
        inSigSignal  =pf.Signal(inSig, sampling_rate=self.fs)
        outputA = self.biquadA.process(inSigSignal)
        outputB = self.biquadB.process(inSigSignal)
        
        outSig = outputA.time * self.fadeOut + outputB.time * self.fadeIn

        self.updateSOS()
        
        return outSig.transpose()
    
    def setSOS(self, sosMatrix):
        """Set new SOS coefficients."""
        assert sosMatrix.shape[1] == 6, "Size of sosMatrix should be (N, 6)."
        assert np.all(sosMatrix[:, 3] == 1), "Leading denominator coefficient should be 1."
        
        if self.sosMatrixNew.shape[0] != sosMatrix.shape[0]:
            self.sosMatrixNew = sosMatrix
            self.sosMatrixLast = sosMatrix
            self.biquadA = pf.FilterSOS(self.sosMatrixLast,sampling_rate=self.fs)#
            self.biquadB = pf.FilterSOS(self.sosMatrixNew,sampling_rate=self.fs)
            self.biquadA.init_state((1,))
            self.biquadB.init_state((1,)) 
        elif np.all(self.sosMatrixNew == [[1, 0, 0, 1, 0, 0], [1, 0, 0, 1, 0, 0]]):
            self.sosMatrixNew = sosMatrix
            self.sosMatrixLast = sosMatrix
        else:
            self.sosMatrixNew = sosMatrix
    
    def updateSOS(self):
        """Update filters by swapping coefficients."""
        temp = self.biquadA
        self.biquadA = self.biquadB
        self.biquadB = temp
        self.sosMatrixLast = self.sosMatrixNew
    

if __name__ == "__main__":
    import scipy.signal as signal
    from Config import Config

    # Initialize config and DSP object
    config = Config()
    config.blockSize = 1024

    # Create input signal
    numberOfBlocks = 30
    signalLength = config.blockSize * numberOfBlocks
    inputSignal = np.random.randn(signalLength, 1)

    # Create filter coefficients for each block
    numberOfSOS = 2
    cutoffFrequencies = np.linspace(0.1, 0.7, numberOfBlocks)  # Changing cutoff frequencies
    sos_filters = np.zeros((numberOfBlocks, numberOfSOS, 6))  # Store SOS coefficients

    variableSOS = VariableSOS_DSP(config)

    for i in range(numberOfBlocks):
        b, a = signal.butter(2 * numberOfSOS, cutoffFrequencies[i], btype='low', analog=False)
        sos_filters[i] = signal.tf2sos(b, a)  # Convert to SOS format

    # Process the signal blockwise
    outputSignal = np.zeros((signalLength, 1))
    for i in range(numberOfBlocks):
        variableSOS.setSOS(sos_filters[i])  # Set the SOS filter for current block
        blockIndex = np.arange(i * config.blockSize, (i + 1) * config.blockSize)
        block = inputSignal[blockIndex]
        outputSignal[blockIndex] = variableSOS.process(block)

    # Plot the input and output signals
    plt.figure(figsize=(10, 4))
    plt.plot(inputSignal, label="Input Signal", alpha=0.7)
    plt.plot(outputSignal, label="Output Signal", linestyle="dashed", alpha=0.9)
    plt.legend()
    plt.xlabel("Time [samples]")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.title("Blockwise Filtering with Variable SOS")
    plt.show()


