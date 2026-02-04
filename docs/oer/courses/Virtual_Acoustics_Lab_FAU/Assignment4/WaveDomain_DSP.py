import numpy as np
import matplotlib.pyplot as plt
from DSP import DSP

class WaveDomain_DSP(DSP):
    """
    FDTD 2D simulation
    Part of this code is adapted from Brian Hamilton's FDTD tutorial for 180th ASA meeting - Matlab code, Copyright 2021
    """
    
    def __init__(self, config):
        super().__init__()
        self.numberOfInputs = 1
        self.numberOfOutputs = 1
        
        # Physical parameters
        self.c = 343  # speed of sound m/s (20degC)
        self.fmax = 1000  # Hz
        self.PPW = 6  # points per wavelength at fmax
        
        # Room configuration
        self.roomSize = config.roomSize[:2]  # box dims (with lower corner at origin) in m
        self.xyIn = np.array(self.roomSize) / 2  # Source input position in m
        self.xyOut = np.array(self.roomSize) / 3  # Receiver output position in m
        
        # Visualization settings
        self.draw = False  # to plot or not
        self.record = False  # to record a video or not
        self.apply_rigid = True  # apply rigid boundaries
        self.hVideo = None  # handle of video plot
        self.videoFileName = 'fdtdSimulation.avi'
        self.umax = 0  # plot range, internal
        
        # Grid parameters
        self.dt = None  # time step (in seconds)
        self.dx = None  # grid spacing (in m)
        
        # Calculate grid spacing, time step, sample rate
        self.dx = self.c / self.fmax / self.PPW  # grid spacing
        # Use slightly more conservative time step for stability with KMap scheme
        self.dt = np.sqrt(0.5) * self.dx / self.c  # Conservative factor for stability
        
        print(f'sample rate = {1/self.dt:.3f} Hz')
        print(f'Δx = {self.dx:.5f} m')
        
        self.checkConfig()
    
    def setPosition(self, sourcePosition, listenerPosition):
        """Set source and listener positions"""
        self.xyIn = np.array(sourcePosition)[:2]
        self.xyOut = np.array(listenerPosition)[:2]
    
    def drawRoom(self, u1g, in_mask, xv, yv):
        """Draw the room visualization"""
        if not hasattr(self, '_draw_mask'):
            # A mask convenient for plotting
            self._draw_mask = np.full_like(in_mask, np.nan)
            self._draw_mask[in_mask] = 1
        
        # Apply mu-law compression for better visualization
        mu = 255  # Standard mu-law parameter
        u_compressed = np.sign(u1g) * np.log1p(mu * np.abs(u1g)) / np.log1p(mu)

        u_normalized = u_compressed
        if not hasattr(self, 'hVideo') or self.hVideo is None:  # setup figure
            plt.figure(figsize=(10, 8))
            u_draw = (u_compressed * self._draw_mask).T
            self.hVideo = plt.imshow(u_draw, extent=[xv[0], xv[-1], yv[0], yv[-1]], 
                                   origin='lower', cmap='turbo', aspect='equal',
                                   vmin=-1, vmax=1)
            plt.xlabel('x')
            plt.ylabel('y')
            plt.colorbar(label='Pressure (normalized, mu-law compressed)')
            plt.xlim([xv[0], xv[-1]])
            plt.ylim([yv[0], yv[-1]])
        else:  # update only plot value
            u_draw = (u_compressed * self._draw_mask).T
            self.hVideo.set_array(u_draw)
            
            # Only draw and pause if interactive drawing is enabled
            if self.draw:
                plt.draw()
                plt.pause(0.001)
    
    @staticmethod
    def createInteriorMask(xv, yv):
        """
        Create interior mask for the room
        
        Args:
            xv, yv: x and y coordinates for the rectangular grid
            
        Returns:
            inMask: mask for 'interior' points
        """
        X, Y = np.meshgrid(xv, yv, indexing='ij')
        inMask = np.zeros(X.shape, dtype=bool)
        inMask[(X >= 0) & (Y >= 0) & (X < np.max(xv)) & (Y < np.max(yv))] = True
           
        return inMask
    
    @staticmethod
    def computeNumberOfNeighbors(inMask):
        """
        Compute number of neighbors for each interior point
        
        Args:
            inMask: mask for 'interior' points
            
        Returns:
            KMap: number of interior neighbors (0 <= KMap <= 4)
        """
        Nx, Ny = inMask.shape
        iX = slice(1, Nx-1)
        iY = slice(1, Ny-1)
        KMap = np.zeros((Nx, Ny))
        
        # Count number of neighbors - exact MATLAB implementation
        # MATLAB: iX = 2:Nx-1, iY = 2:Ny-1
        # MATLAB: KMap(iX,iY) = inMask(iX+1,iY) + inMask(iX-1,iY) + inMask(iX,iY+1) + inMask(iX,iY-1)
        # Convert boolean mask to int for proper arithmetic
        inMask_int = inMask.astype(int)
        KMap[iX, iY] = (inMask_int[2:Nx, 1:Ny-1] + inMask_int[0:Nx-2, 1:Ny-1] + 
                       inMask_int[1:Nx-1, 2:Ny] + inMask_int[1:Nx-1, 0:Ny-2])
        KMap[~inMask] = 0
        
        return KMap
    
    def recordVideo(self, filename):
        """Record video frames"""
        if filename is None:
            # Close the writer object
            if hasattr(self, '_writer'):
                self._writer.finish()
                delattr(self, '_writer')
            return
        
        # Check if we need to create a new writer (first time or filename changed)
        if not hasattr(self, '_writer') or not hasattr(self, '_current_filename') or self._current_filename != filename:
            # Close existing writer if it exists
            if hasattr(self, '_writer'):
                self._writer.finish()
                delattr(self, '_writer')
            
            # Create the video writer with 60 fps
            from matplotlib.animation import FFMpegWriter, PillowWriter
            
            # Determine format based on file extension
            if filename.lower().endswith('.gif'):
                self._writer = PillowWriter(fps=60)
            else:
                self._writer = FFMpegWriter(fps=60)
            
            self._writer.setup(plt.gcf(), filename)
            self._current_filename = filename
        
        self._writer.grab_frame()  # convert the image to a frame
    
    def resetVideoRecording(self):
        """Reset video recording state - useful when changing filenames"""
        if hasattr(self, '_writer'):
            self._writer.finish()
            delattr(self, '_writer')
        if hasattr(self, '_current_filename'):
            delattr(self, '_current_filename')
        if hasattr(self, 'hVideo'):
            delattr(self, 'hVideo')
        if hasattr(self, '_draw_mask'):
            delattr(self, '_draw_mask')
        # Close any existing matplotlib figures
        import matplotlib.pyplot as plt
        plt.close('all')
