
import numpy as np

class RT60:
    """RT60 reverberation time configuration"""
    def __init__(self):
        self.Low = 2.0
        self.High = 0.5

class Late:
    """Late reverberation configuration"""
    def __init__(self):
        self.numberOfDelays = 8
        self.feedbackMatrix = np.eye(8)  # Identity matrix by default
        self.MaximumDelay = 48000

class Early:
    """Early reflection configuration"""
    def __init__(self):
        self.maxImageSourceOrder = 3
        self.MaximumDelay = 48000

class Config:
    def __init__(self):
        # default config
        self.fs = 48000
        self.speedOfSound = 343  # ToDo Check pyFar constants
        self.blockSize = 256
        self.temperature = 20
        self.relativeHumidity = 0.5
        self.headRadius = 0.09
        self.roomSize = np.array([10, 7, 4])
        self.spatialEncoding = 'object'
        self.ambiOrder = 3
        self.maxre = True
        self.spatialDecoding = 'binaural'
        # self.lsPositions = self.defaultLoudspeakers()
        self.maximumDelay = self.fs
        
        # Nested configurations
        self.RT60 = RT60()
        self.Early = Early()
        self.Late = Late()

    def defaultLoudspeakers(self):
        # Define loudspeaker layout
        # The loudspeaker layout consists of 3 concentric rings, plus a top
        # louspeaker (also called "voice of god"). As a loudspeaker directly under
        # the listener is hard to implement, only the three main layers will be
        # symmetric. The coordinates are given in azimuth and elevation, and will
        # be then translated to Cartesian coordinates.
        
        # Define Loudspeaker first
        layerNum = [1, 12, 8, 8]
        layerEle = [90, 0, 45, -45]
        azi = {}
        ele = {}
        for it in range(0, len(layerNum)):
            azi[it] = np.linspace(0, 360 - 360/layerNum[it], layerNum[it])
            ele[it] = layerEle[it] * np.ones(len(azi[it]))

        lsAzi = azi
        # lsEle = [ele{:}]
        
        # [lsX, lsY, lsZ] = sph2cart(deg2rad(lsAzi), deg2rad(lsEle), 1)
        # lsPositions = [lsX.', lsY.', lsZ.']
        # return lsPositions

