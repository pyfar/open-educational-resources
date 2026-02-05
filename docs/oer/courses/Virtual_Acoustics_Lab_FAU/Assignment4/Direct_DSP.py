import numpy as np
from DSP import DSP
from VariableDelay_DSP import VariableDelay_DSP
from VariableSOS_DSP import VariableSOS_DSP
from air_absorption_iso import air_absorption_iso
from design_one_pole_filter import design_one_pole_filter
from cart2sph_vec import cart2sph_vec
from DSP import m2smp, db2mag

class Direct_DSP(DSP):
    """
    Direct sound filter including propagation delay and air absorption.
    """
    
    def __init__(self, config):
        """
        Initialize Direct_DSP.
        
        Args:
            config: Configuration object
        """
        self.speedOfSound = config.speedOfSound
        self.fs = config.fs
        self.blockSize = config.blockSize
        
        # Initialize relative positions (2D)
        self.lastRelativePosition = np.array([1, 1])
        self.relativePosition = np.array([1, 1])
        
        # Define input/output channels
        self.numberOfInputs = 1
        self.numberOfOutputs = 1
        
        # Initialize processing blocks
        self.propagationDelay = VariableDelay_DSP(1, config)
        self.airAbsorption = VariableSOS_DSP(config)
        
        # Precompute air absorption
        f = np.linspace(0, self.fs/2, 2**10)
        T = config.temperature
        hr = config.relativeHumidity
        self.airAbsorptionAlpha, _ = air_absorption_iso(f, T, hr)
        self.airAbsorptionFrequency = f
        
        self.checkConfig()
    
    def process(self, inSig):
        """
        Process input signal through direct path.
        
        Args:
            inSig (np.ndarray): Input signal
            
        Returns:
            np.ndarray: Processed output signal
        """
        # Calculate distances
        lastDistance = np.linalg.norm(self.lastRelativePosition, 2)
        distance = np.linalg.norm(self.relativePosition, 2)
        
        # Apply propagation delay
        delay_samples = m2smp(distance, self.speedOfSound, self.fs)
        self.propagationDelay.setDelay(np.array([[delay_samples]]))
        outSig = self.propagationDelay.process(inSig)
        
        # Apply air absorption
        scaledAlpha = db2mag(self.airAbsorptionAlpha * distance)
        sos = design_one_pole_filter(scaledAlpha[[0, -1]], scaledAlpha[[0, -1]])
        sos = np.ascontiguousarray(sos.T)  # Transpose to get correct shape (N, 6)
        self.airAbsorption.setSOS(sos)
        outSig = self.airAbsorption.process(outSig)
        
        # Apply distance gain
        distances = np.linspace(lastDistance, distance, self.blockSize)
        distanceGain = np.minimum(1 / distances, 1)
        outSig = outSig * distanceGain[:, np.newaxis]
        
        # Update last relative position to current
        self.lastRelativePosition = self.relativePosition.copy()
        
        return outSig
    
    def getDoa(self):
        """
        Return the direction of arrival (DOA) in spherical coordinates [Azimuth, Elevation, Radius].
        
        Returns:
            np.ndarray: DOA in spherical coordinates
        """
        return cart2sph_vec(self.relativePosition)
    
    def setScene(self, scene):
        """
        Compute the relative position as the difference between source and listener positions.
        
        Args:
            scene: Scene object containing source and listener positions
        """
        self.relativePosition = np.array(scene.sourcePosition) - np.array(scene.listenerPosition) 