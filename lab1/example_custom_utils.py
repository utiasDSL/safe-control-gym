"""Example utility module.

Please use a file like this one to add extra functions.

"""
import numpy as np

def exampleFunction():
    """Example of user-defined function.

    """
    x = -1
    return x

def generateCircle(radius, start_point, num_points):
    """
    Generates a numpy array of 3D coordinates forming a circle based on a starting point on its edge.

    Args:
        radius (float): Radius of the circle.
        start_point (tuple of float): (x, y, z) coordinates of the starting point on the edge of the circle.
        num_points (int): Number of points forming the circle.

    Returns:
        np.ndarray: A numpy array of shape (num_points, 3) containing the 3D coordinates.
    """
    # Calculate center of the circle assuming the start point is on the edge
    center = (start_point[0] + radius, start_point[1], start_point[2])  # Center is radius units left of start_point

    # Generate the points
    theta = np.linspace(0, 2 * np.pi, num_points, endpoint=False) + np.pi
    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)
    z = np.full_like(x, center[2])  # All points have the same z-coordinate as the center
    return np.column_stack((x, y, z))