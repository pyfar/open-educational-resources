import numpy as np
from DSP import DSP

class AmbiDecoder_DSP(DSP):
    """
    Take Ambisonics Signal and Decode to Loudspeaker signals
    """
    def __init__(self, config):
        """
        Constructor
        
        Args:
            config: Configuration object containing:
                - ambiOrder: Ambisonics order used for decoding
                - maxre: Flag for enabling/disabling max-re weighting
                - lsPositions: Loudspeaker positions in cartesian coordinates
        """
        super().__init__()
        self.config = config
        self.ambiOrder = config.ambiOrder
        self.numAmbiCh = (config.ambiOrder + 1) ** 2
        self.maxre = config.maxre
        self.lsPositions = config.lsPositions
        self.numLs = self.lsPositions.shape[0]
        
        self.decoderMatrix = self.calculateSamplingDecoder()
        
    def process(self, inSig):
        """
        Apply the decoding matrix to the Ambisonics signals to obtain loudspeaker signals
        
        Args:
            inSig (np.ndarray): Input signal matrix [S x (N+1)^2] SH signal matrix
            
        Returns:
            np.ndarray: Output signal matrix [S x numLs] loudspeaker signal matrix
        """
        return inSig @ self.decoderMatrix
    
    def plotLoudspeakerSignals(self, lsSignals):
        """
        Plot the loudspeaker signals and positions
        
        Args:
            lsSignals (np.ndarray): Loudspeaker signals to plot
        """
        import matplotlib.pyplot as plt
        from matplotlib import cm
        
        # Normalize signals
        lsSignals = lsSignals / np.max(np.abs(lsSignals))
        
        # Create figure with two subplots
        fig = plt.figure(figsize=(12, 5))
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        ax2 = fig.add_subplot(1, 2, 2)
        
        # Plot loudspeaker positions
        colors = cm.jet(np.linspace(0, 1, self.numLs))
        scatter = ax1.scatter(self.lsPositions[:, 0], 
                            self.lsPositions[:, 1],
                            self.lsPositions[:, 2],
                            c=colors, marker='s')
        
        # Add labels
        for i in range(self.numLs):
            ax1.text(self.lsPositions[i, 0],
                    self.lsPositions[i, 1],
                    self.lsPositions[i, 2],
                    str(i + 1))
        
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_zlabel('z')
        ax1.set_box_aspect([1,1,1])
        
        lsmax = np.max(np.abs(self.lsPositions))
        ax1.set_xlim([-lsmax, lsmax])
        ax1.set_ylim([-lsmax, lsmax])
        ax1.set_zlim([-lsmax, lsmax])
        
        # Plot loudspeaker signals
        for i in range(self.numLs):
            ax2.plot(lsSignals[:, i] + 2*i, color=colors[i])
        
        ax2.set_yticks(np.arange(1, self.numLs*2, 2))
        ax2.set_yticklabels(range(1, self.numLs + 1))
        ax2.grid(True)
        ax2.set_xlabel('Time (samples)')
        ax2.set_ylabel('Loudspeaker')
        ax2.set_title('Loudspeaker Signals')
        
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)