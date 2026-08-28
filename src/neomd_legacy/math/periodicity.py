import numpy as np


def minimum_image_coordinates(coord_ref, coords, box):
    """
    Calculate the nearest coordinates of particles relative to a reference
    particle under periodic boundary conditions.

    Parameters:
    coord_ref (numpy array, shape (3)): Coordinates of the reference particle.
    coords (numpy array, shape (n, 3)): Coordinates of the particles to adjust.
    box (numpy array, shape (3)): The dimensions of the periodic box [Lx, Ly, Lz].

    Returns:
    numpy array: Adjusted coordinates of the particles under periodic constraints.
    """
    # Calculate the displacement vectors between the reference particle and other particles
    displacement = coords - coord_ref
    displacement -= box * np.round(displacement / box)

    # Compute the adjusted coordinates
    adjusted_coords = coord_ref + displacement
    return adjusted_coords
