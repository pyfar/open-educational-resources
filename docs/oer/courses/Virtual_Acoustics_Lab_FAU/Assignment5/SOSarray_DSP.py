import numpy as np
from scipy.signal import sosfilt, sosfilt_zi
from DSP import DSP

class SOSarray_DSP(DSP):
    """
    K parallel biquad filters using scipy directly 
    """
    
    def __init__(self, numberOfChannels, config):
        """
        Initialize multiple parallel SOS filters
        
        Args:
            numberOfChannels: Number of parallel channels
            config: Configuration object
        """
        self.numberOfChannels = numberOfChannels
        self.sos_coeffs = [np.array([[1, 0, 0, 1, 0, 0]]) for _ in range(numberOfChannels)]
        self.states = [None for _ in range(numberOfChannels)]
        
        self.numberOfInputs = numberOfChannels
        self.numberOfOutputs = numberOfChannels
        self.blockSize = config.blockSize
        self.checkConfig()
    
    def process(self, input_signal):
        """
        Process input signal through parallel SOS filters
        
        Args:
            input_signal: Input signal of shape (blockSize, numberOfChannels)
        
        Returns:
            output: Filtered output of shape (blockSize, numberOfChannels)
        """
        output = np.zeros_like(input_signal)
        
        for it in range(self.numberOfChannels):
            # Get input for this channel as contiguous array
            x = np.ascontiguousarray(input_signal[:, it])
            
            # Initialize state if needed
            if self.states[it] is None:
                self.states[it] = sosfilt_zi(self.sos_coeffs[it]) * 0
            
            # Ensure state is contiguous
            self.states[it] = np.ascontiguousarray(self.states[it])
            
            # Filter with state
            y, self.states[it] = sosfilt(self.sos_coeffs[it], x, zi=self.states[it])
            output[:, it] = y
        
        return output
    
    def setSOS(self, sosmatrix):
        """
        Set SOS coefficients for all filters
        
        Args:
            sosmatrix: SOS coefficients of shape (6, 1, numberOfChannels)
        """
        for it in range(self.numberOfChannels):
            # Extract coefficients for this channel
            if sosmatrix.ndim == 3:
                if sosmatrix.shape[1] == 1:
                    # (6, 1, N) format - reshape to (1, 6) for single biquad
                    sos_ch = sosmatrix[:, 0, it].copy().reshape(1, 6)
                else:
                    # (numberOfSOS, 6, N) format
                    sos_ch = sosmatrix[:, :, it].copy()
            else:
                # (6, N) format
                sos_ch = sosmatrix[:, it].copy().reshape(1, 6)
            
            # Ensure SOS coefficients are C-contiguous
            self.sos_coeffs[it] = np.ascontiguousarray(sos_ch)
            # Reset state when coefficients change and ensure contiguity
            self.states[it] = np.ascontiguousarray(sosfilt_zi(self.sos_coeffs[it]) * 0)


if __name__ == "__main__":
    from Config import Config
    import matplotlib.pyplot as plt
    
    # Initialize config
    config = Config()
    config.blockSize = 256
    
    # Create multi-channel filter
    numberOfChannels = 3
    sosArray = SOSarray_DSP(numberOfChannels, config)
    
    # Create test SOS filters
    sos_test = np.zeros((6, 1, numberOfChannels))
    sos_test[0, 0, :] = [0.5, 0.6, 0.7]  # b0
    sos_test[3, 0, :] = 1.0  # a0
    sos_test[4, 0, :] = [0.5, 0.4, 0.3]  # a1
    
    sosArray.setSOS(sos_test)
    
    # Create test signal
    numBlocks = 10
    signalLength = config.blockSize * numBlocks
    signal = np.zeros((signalLength, numberOfChannels))
    signal[0, :] = 1  # Impulse
    
    output = np.zeros_like(signal)
    
    for it_block in range(numBlocks):
        block_index = slice(it_block * config.blockSize, (it_block + 1) * config.blockSize)
        block = signal[block_index, :]
        output[block_index, :] = sosArray.process(block)
    
    # Plot results
    plt.figure(figsize=(12, 6))
    for ch in range(numberOfChannels):
        plt.subplot(numberOfChannels, 1, ch + 1)
        plt.plot(output[:, ch])
        plt.title(f'Channel {ch + 1}')
        plt.grid(True)
    plt.tight_layout()
    plt.show()

