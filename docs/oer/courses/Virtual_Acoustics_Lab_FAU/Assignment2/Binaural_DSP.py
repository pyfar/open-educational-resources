import numpy as np
import sofar as sf
import matplotlib.pyplot as plt
from scipy.signal import lfilter, resample_poly
import pyfar as pf


from hrirsDiffuseFieldEQ import hrirsDiffuseFieldEQ
from BlockConvolver_DSP import BlockConvolverDSP as  BlockConvolverDSP
from DSP import DSP

def sph2cart(azimuth, elevation, r=1):
    x = r * np.cos(elevation) * np.cos(azimuth)
    y = r * np.cos(elevation) * np.sin(azimuth)
    z = r * np.sin(elevation)
    return np.column_stack((x, y, z))


class BinauralDSP(DSP):
    def __init__(self, config, num_src):
        self.blockSize = config.blockSize
        self.num_src = num_src
        self.doa_azi = None
        self.doa_ele = None
        
        self.numberOfInputs = self.num_src
        self.numberOfOutputs = 2
        
        # Load SOFA data
        self.sofa_data = sf.read_sofa('FABIAN.sofa')
        self.num_hrir = self.sofa_data.get_dimension('M')
        self.hrirs = self.sofa_data.Data_IR
        self.fs = self.sofa_data.Data_SamplingRate
        
        hrir_length = self.sofa_data.get_dimension('N')
        new_length = np.ceil(hrir_length * config.fs / self.fs).astype(int)
        # Resample HRIRs if sampling rates do not match
        if self.fs != config.fs:
            self.hrirs = resample_poly(self.hrirs, config.fs, self.fs, axis=2)
            self.fs = config.fs
        assert self.fs == config.fs

        # Convert HRIR positions to Cartesian coordinates
        self.hrir_positions = sph2cart(
            np.deg2rad(self.sofa_data.SourcePosition[:, 0]),
            np.deg2rad(self.sofa_data.SourcePosition[:, 1])
        )
        
        # Compute diffuse field EQ
        grid_weights = np.ones(self.num_hrir) * (4 * np.pi / self.num_hrir)
        diff_eq_taps = hrirsDiffuseFieldEQ(self.hrirs, True, grid_weights)

        # Apply diffuse equalization
        for j in range(self.sofa_data.get_dimension('R')): 
            for k in range(self.num_hrir):  # Loop over HRIRs
                x = self.hrirs[k, j, :].astype(np.float64)
                b = diff_eq_taps.astype(np.float64)
                self.hrirs[k, j, :] = lfilter(b, [1.0], x, axis=0)

        # Initialize convolvers
        if self.num_src:
            self.convolver_left = BlockConvolverDSP(self.blockSize, self.numberOfInputs, self.hrirs.shape[2])
            self.convolver_right = BlockConvolverDSP(self.blockSize, self.numberOfInputs, self.hrirs.shape[2])
            self.checkConfig()

    def set_doa(self, azi, ele):
        assert azi.shape == (self.numberOfInputs, )
        assert ele.shape == (self.numberOfInputs, )
        
        self.doa_azi = azi
        self.doa_ele = ele  
        idx_hrir = self.nearestPoint(azi, ele)
        
        irs_left = self.hrirs[idx_hrir, 0, :].transpose()
        irs_right = self.hrirs[idx_hrir, 1, :].transpose()
        
        self.convolver_left.setIRs(irs_left)
        self.convolver_right.setIRs(irs_right)
