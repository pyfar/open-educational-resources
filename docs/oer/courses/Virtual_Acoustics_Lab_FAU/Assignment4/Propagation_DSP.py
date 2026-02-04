import numpy as np
from DSP import DSP
from VariableDelay_DSP import VariableDelay_DSP
from VariableSOS_DSP import VariableSOS_DSP
from air_absorption_iso import air_absorption_iso
from design_one_pole_filter import design_one_pole_filter
from DSP import m2smp, db2mag

class Propagation_DSP(DSP):
    """
    Includes delay, gain, air and reflections absorption.
    """
    
    def __init__(self, config):
        """
        Initialize Propagation_DSP.
        
        Args:
            config: Configuration object
        """
        self.propagationDelay = VariableDelay_DSP(1, config)
        self.airAbsorption = VariableSOS_DSP(config)
        self.reflectionAbsorption = VariableSOS_DSP(config)
        
        self.sourceDistance = 1
        self.fs = config.fs
        self.speedOfSound = config.speedOfSound
        
        # Precompute air absorption
        f = np.linspace(0, config.fs/2, 2**10)
        T = config.temperature
        hr = config.relativeHumidity
        self.alphaISO, _ = air_absorption_iso(f, T, hr)
        
        self.numberOfInputs = 1
        self.numberOfOutputs = 1
        self.blockSize = config.blockSize
        self.checkConfig()
    
    def process(self, input_signal):
        """
        Process input signal through propagation path.
        
        Args:
            input_signal (np.ndarray): Input signal
            
        Returns:
            np.ndarray: Processed output signal
        """
        # Delay, based on obj.sourceDistance
        delay_samples = m2smp(self.sourceDistance, self.speedOfSound, self.fs)
        self.propagationDelay.setDelay(np.array([[delay_samples]]))
        output = self.propagationDelay.process(input_signal)
        
        # Air Absorption, based on obj.sourceDistance
        sos_air = self.computePropagationAttenuation(self.sourceDistance)
        self.airAbsorption.setSOS(sos_air)
        output = self.airAbsorption.process(output)
        
        # Reflection Absorption (not implemented in this version)
        # self.setReflectionAbsorption(sos_refl)
        # output = self.reflectionAbsorption.process(output)
        
        return output
    
    def setDistance(self, sourceDistance):
        """
        Set source distance, used to calculate propagationDelay and propagationAttenuation.
        
        Args:
            sourceDistance (float): Distance in meters
        """
        self.sourceDistance = sourceDistance
    
    def setAirAbsorption(self, sos):
        """
        Set air absorption filter.
        
        Args:
            sos (np.ndarray): SOS coefficients
        """
        self.airAbsorption.setSOS(sos)
    
    def setReflectionAbsorption(self, sos):
        """
        Set reflection absorption filter.
        
        Args:
            sos (np.ndarray): SOS coefficients
        """
        self.reflectionAbsorption.setSOS(sos)
    
    def computePropagationAttenuation(self, distance):
        """
        Compute air absorption and distance gain; return sos filter.
        
        Args:
            distance (float): Distance in meters
            
        Returns:
            np.ndarray: SOS filter coefficients
        """
        # Scale alpha by distance
        scaledAlpha = db2mag(self.alphaISO[[0, -1]] * distance)
        sos = design_one_pole_filter(scaledAlpha, scaledAlpha)
        
        # Apply distance gain (handle zero distance)
        if distance > 0:
            distanceGain = min(1 / distance, 1)
        else:
            distanceGain = 1  # No attenuation for zero distance
        sos[0:3, 0] = sos[0:3, 0] * distanceGain
        
        # Transpose to get correct shape (N, 6) and ensure C-contiguous
        sos = np.ascontiguousarray(sos.T)
        
        return sos 