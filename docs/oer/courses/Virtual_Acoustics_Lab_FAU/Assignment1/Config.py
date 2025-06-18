
import numpy as np

class Config:
  def __init__(self):
    # default config
    self.fs = 48000
    self.speedOfSound = 343 #ToDo Check pyFar constants
    self.blockSize = 256
    self.temperature = 20
    self.relativeHumidity = 0.5
    self.headRadius = 0.09
    self.roomSize = [10, 7, 4]
    self.spatialEncoding = 'selfect'
    self.ambiOrder = 3
    self.maxre = True
    self.spatialDecoding = 'binaural'
    #self.lsPositions = self.defaultLoudspeakers()
    self.maximumDelay = self.fs
    self.RT60_Low = 2
    self.RT60_High = 0.5
    
    self.Early_maxImageSourceOrder = 3
    self.Early_MaximumDelay = self.fs
    
    self.Late_numberOfDelays = 8
    self.Late_feedbackMatrix = np.ones(self.Late_numberOfDelays)
    self.Late_MaximumDelay = self.fs

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
        azi[it] = np.linspace(0,360 - 360/layerNum[it],layerNum[it])
        ele[it] = layerEle[it] * np.ones(len(azi[it]))

    lsAzi = azi
    #lsEle = [ele{:}]
    
    #[lsX, lsY, lsZ] = sph2cart(deg2rad(lsAzi), deg2rad(lsEle), 1)
    #lsPositions = [lsX.', lsY.', lsZ.']
    #return lsPositions

