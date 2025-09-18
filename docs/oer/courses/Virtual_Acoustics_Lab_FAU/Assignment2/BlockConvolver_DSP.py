import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as fft

class BlockConvolverDSP:
    """
    Overlap-add convolution with variable filters.
    """
    def __init__(self, block_size, num_channels, ir_length):
        """
        Initialize the BlockConvolverDSP.
        """
        self.block_size = block_size
        self.ir_length = ir_length
        self.number_of_inputs = num_channels
        self.number_of_outputs = num_channels
        
        # Initialize with zeros
        self.setIRs(np.zeros((ir_length, num_channels )))
        self.overlap = np.zeros((self.ir_length - 1, self.number_of_inputs))
    
    def setIRs(self, irs):
        """
        Set new filters.
        """
        assert irs.shape == (self.ir_length, self.number_of_inputs), "Check initialization"
        self.irs = irs
        
        # Pre-compute frequency domain representation
        nfft = self.block_size + self.ir_length - 1
        self.H = fft.fft(self.irs, nfft, axis=0)
    
    def process(self, input_block):
        """
        Process input signal using overlap-add convolution.
        """
        assert input_block.shape == (self.block_size, self.number_of_inputs), "Input shape mismatch"
        
        nfft = self.block_size + self.ir_length - 1
        
        # Zero-padded input spectrum
        X = fft.fft(input_block, nfft, axis=0)
        
        # Convolution in frequency domain
        Y = self.H * X
        
        # Back to time-domain
        y = np.fft.irfft(Y, nfft, axis=0)
        
        # Overlap-add
        current_out = y + np.vstack((self.overlap, np.zeros((self.block_size, self.number_of_inputs))))
        
        # Extract block output
        block_output = current_out[:self.block_size, :]
        
        # Save new overlap
        self.overlap = current_out[self.block_size:, :]
        
        return block_output
