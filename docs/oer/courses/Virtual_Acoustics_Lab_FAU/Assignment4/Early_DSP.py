import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from DSP import DSP
from Propagation_DSP import Propagation_DSP
from cart2sph_vec import cart2sph_vec

class Early_DSP(DSP):
    """
    Image Sources like early reflections.
    """

    def __init__(self, config):
        """
        Initialize Early_DSP.

        Args:
            config: Configuration object
        """
        self.fs = config.fs
        self.speedOfSound = 343

        self.sourcePosition = np.array([0, 0])
        self.listenerPosition = np.array([0, 0])
        self.imageSourcePosition = np.array([0, 0])

        self.roomSize = config.roomSize

        self.maxImageSourceOrder = config.Early_maxImageSourceOrder
        self.imageSourceList = self.prepareImageSources(self.maxImageSourceOrder)
        numberOfImageSources = self.imageSourceList.shape[0]

        # The process function receives one channel and returns one
        # channel per image sources as output.
        self.numberOfInputs = 1
        self.numberOfOutputs = numberOfImageSources

        # Each image source is processed as one propagation path
        self.propagations = []
        for it in range(numberOfImageSources):
            self.propagations.append(Propagation_DSP(config))

        # Initial computation of image source positions
        # self.setPosition(self.sourcePosition, self.listenerPosition)

        self.blockSize = config.blockSize

    def getDoa(self):
        """
        Get direction of arrival for all image sources.

        Returns:
            np.ndarray: DOA in spherical coordinates
        """
        # Hint: Use cart2sphVec
        relativePosition = self.imageSourcePosition - self.listenerPosition
        doa = cart2sph_vec(relativePosition)
        return doa

    def getDistance(self):
        """
        Get distances to all image sources.

        Returns:
            np.ndarray: Distances to image sources
        """
        # Hint: Use vecnorm
        relativePosition = self.imageSourcePosition - self.listenerPosition
        distance = np.linalg.norm(relativePosition, axis=1)
        return distance

    def drawShoeboxRoom(self, ax, width, depth):
        """
        Draw a 2D shoebox room using a rectangle.

        Args:
            ax: matplotlib 2D axis object
            width: room width (x-dimension)
            depth: room depth (y-dimension)
        """
        # Draw room as a rectangle
        from matplotlib.patches import Rectangle
        room = Rectangle((0, 0), width, depth, 
                        fill=True, facecolor='lightgray', 
                        edgecolor='black', linewidth=2, alpha=0.3)
        ax.add_patch(room)

    def plotImageSources(self):
        """
        2D plot of the image sources of a shoebox.
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        # Draw room boundaries
        self.drawShoeboxRoom(ax, self.roomSize[0], self.roomSize[1])

        # Plot image sources
        ax.scatter(self.imageSourcePosition[:, 0],
                  self.imageSourcePosition[:, 1],
                  c='black', marker='x', s=50, label='Image Sources')

        # Plot source
        ax.scatter(self.sourcePosition[0],
                  self.sourcePosition[1],
                  c='red', marker='*', s=200, label='Source', edgecolors='black', zorder=5)

        # Plot receiver
        ax.scatter(self.listenerPosition[0],
                  self.listenerPosition[1],
                  c='green', marker='o', s=100, label='Receiver', edgecolors='black', zorder=5)

        ax.set_xlabel('x in m')
        ax.set_ylabel('y in m')
        ax.set_title('Image Sources')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.show()

    def animationImageSources(self, use_animation=False, save_gif=False):
        """
        2D animation of the image sources waves.

        Args:
            use_animation (bool): If True, creates animated plot. If False, shows static frames.
            save_gif (bool): If True, saves animation as GIF file.
        """
        # All image sources are already 2D, so use them directly
        ISPos = self.imageSourcePosition[:, :2]
        # Add real source at the beginning
        ISPos = np.vstack([self.sourcePosition[:2], ISPos])

        if use_animation:
            # Proper animation for interactive environments
            return self._create_animation(ISPos, save_gif)
        else:
            # Static frames for Jupyter notebooks
            return self._show_static_frames(ISPos)

    def _create_animation(self, ISPos, save_gif=False):
        """Create proper matplotlib animation"""
        fig, ax = plt.subplots(figsize=(10, 6))
        room = plt.Rectangle((0, 0), self.roomSize[0], self.roomSize[1],
                           fill=False, color='black', linewidth=2)
        ax.add_patch(room)

        numberOfSources = ISPos.shape[0]
        circles = []
        for it in range(numberOfSources):
            circle = plt.Circle((0, 0), 0, fill=False, color='blue', alpha=0.6)
            circles.append(circle)
            ax.add_patch(circle)

        # Add source markers
        # ax.scatter(ISPos[:, 0], ISPos[:, 1], c='red', s=50, marker='o',
        #           label='Image Sources', zorder=5)

        # Add source and receiver position markers
        ax.scatter(self.sourcePosition[0], self.sourcePosition[1],
                  c='red', s=30, marker='*', label='Source', zorder=10, edgecolors='black')
        ax.scatter(self.listenerPosition[0], self.listenerPosition[1],
                  c='green', s=30, marker='o', label='Receiver', zorder=10, edgecolors='black')

        ax.set_xlim(0, self.roomSize[0])
        ax.set_ylim(0, self.roomSize[1])
        ax.set_aspect('equal')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_title('2D Image Source Wave Animation')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        def animate(frame):
            r = frame / 10
            for it in range(numberOfSources):
                xy = ISPos[it, :]
                circles[it].center = xy
                circles[it].radius = r
            return circles

        anim = FuncAnimation(fig, animate, frames=200, interval=50, blit=False, repeat=True)

        if save_gif:
            try:
                anim.save('image_source_animation.gif', writer='pillow', fps=20)
                print("Animation saved as 'image_source_animation.gif'")
            except Exception as e:
                print(f"Could not save GIF: {e}")

        plt.show()
        return anim

    def _show_static_frames(self, ISPos):
        """Show static frames for Jupyter notebooks"""
        numberOfSources = ISPos.shape[0]

        # Create multiple subplots showing different time instances
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        time_frames = [0, 20, 40, 60, 80, 100]

        for idx, frame in enumerate(time_frames):
            ax = axes[idx]

            # Draw room
            room = plt.Rectangle((0, 0), self.roomSize[0], self.roomSize[1],
                               fill=False, color='black', linewidth=2)
            ax.add_patch(room)

            # Draw wavefronts
            r = frame / 10
            for it in range(numberOfSources):
                xy = ISPos[it, :]
                if r > 0:
                    circle = plt.Circle(xy, r, fill=False, color='blue', alpha=0.6)
                    ax.add_patch(circle)

            # Add source markers
            ax.scatter(ISPos[:, 0], ISPos[:, 1], c='red', s=30, marker='o', zorder=5)

            ax.set_xlim(-1, self.roomSize[0] + 1)
            ax.set_ylim(-1, self.roomSize[1] + 1)
            ax.set_aspect('equal')
            ax.set_xlabel('x (m)')
            ax.set_ylabel('y (m)')
            ax.set_title(f'Time: {frame/10:.1f} s')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.suptitle('2D Image Source Wave Propagation (Static Frames)', y=1.02)
        plt.show()

        print("Animation shown as static frames.")
        print("For interactive animation, call: early.animationImageSources(use_animation=True)")
        print("To save as GIF, call: early.animationImageSources(use_animation=True, save_gif=True)")
