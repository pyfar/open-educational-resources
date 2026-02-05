
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
    self.lsPositions = default_loudspeakers()
    self.maximumDelay = self.fs
    self.RT60_Low = 2
    self.RT60_High = 0.5
    
    self.Early_maxImageSourceOrder = 3
    self.Early_MaximumDelay = self.fs
    
    self.Late_numberOfDelays = 8
    self.Late_feedbackMatrix = np.ones(self.Late_numberOfDelays)
    self.Late_MaximumDelay = self.fs

def default_loudspeakers():
    """
    Define a default loudspeaker layout consisting of concentric rings
    and a top speaker (Voice of God). Returns positions in Cartesian coordinates.
    
    Output:
        ls_positions: np.ndarray of shape [num_ls, 3]
    """
    layer_num = [1, 12, 8, 8]
    layer_ele = [90, 0, 45, -45]

    azis = []
    eles = []

    for num, ele in zip(layer_num, layer_ele):
        azi = np.linspace(0, 360 - 360 / num, num)
        el = np.full_like(azi, ele)
        azis.append(azi)
        eles.append(el)

    ls_azi = np.concatenate(azis)
    ls_ele = np.concatenate(eles)

    # Convert degrees to radians
    ls_azi_rad = np.deg2rad(ls_azi)
    ls_ele_rad = np.deg2rad(ls_ele)

    # Convert spherical to Cartesian
    ls_x = np.cos(ls_ele_rad) * np.cos(ls_azi_rad)
    ls_y = np.cos(ls_ele_rad) * np.sin(ls_azi_rad)
    ls_z = np.sin(ls_ele_rad)

    ls_positions = np.stack((ls_x, ls_y, ls_z), axis=-1)

    return ls_positions
