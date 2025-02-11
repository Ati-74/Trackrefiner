"""
draw_image_bw and calculate_colony_density are based on code from Sohaib Nadeem and Jonathan Chalaturnik:
https://github.com/ingallslab/bsim-related/blob/main/bsim_related/data_processing/

Modifications by Aaron Yip:
-Repurposed to work with CellModeller data
-Fixed bugs with colony area calculations
-Added documentation

Instructions:
Call the main() function, giving a cellStates dict as an argument.
Optional: include path to a directory to export plots
"""
# Standard modules
import math
import numpy as np
import pandas as pd
import cv2
from PIL import Image, ImageDraw
import skimage.morphology
import skimage.measure


def convert_cellmodeller_orientation_to_radians(cell_dir):
    """
    Converts cell.dir to an orientation in radians

    @param  cell_dir    list containing cell direction in [x, y, z]
    @return orientation cell orientation in radians (float)

    Note: only valid for 2D case
    """
    # Convert vector into a unit vector
    magnitude = np.linalg.norm(cell_dir)
    cell_dir_unit_vector = cell_dir / magnitude

    # Calculate orientation in radians
    if cell_dir_unit_vector[0] != 0:
        orientation = np.arctan(cell_dir_unit_vector[1] / cell_dir_unit_vector[0])
    else:
        orientation = np.pi / 2

    return orientation


def get_cell_data_to_draw_image_bw(cells, um_pixel_ratio=0.144):
    """
    Convert CellModeller data into a format suitable for drawing black/white images

    @param  cells           cellStates dict
    @param  um_pixel_ratio  micrometer to pixel ratio (0.144 is default for 63X objective on microscope)
    @return ...             see variable names in return statement
    """
    # Add pixels to img border to avoid cutting off cells
    add_pixels_to_img_border = 5

    # Initialize storage variables
    n_cells = len(cells)
    cell_centers_x = np.zeros(n_cells)
    cell_centers_y = np.zeros(n_cells)
    cell_lengths = np.zeros(n_cells)
    cell_radii = np.zeros(n_cells)
    cell_orientations = np.zeros(n_cells)

    # Get cell data
    for i, cell in enumerate(cells.values()):
        cell_centers_x[i] = cell.pos[0]
        cell_centers_y[i] = cell.pos[1]
        cell_lengths[i] = cell.length
        cell_radii[i] = cell.radius
        cell_orientations[i] = convert_cellmodeller_orientation_to_radians(cell.dir)

    # Ensure all cells have positive coordinates for drawing
    min_x = min(cell_centers_x) - add_pixels_to_img_border
    min_y = min(cell_centers_y) - add_pixels_to_img_border
    cell_centers_x = cell_centers_x + abs(min_x)
    cell_centers_y = cell_centers_y + abs(min_y)

    # Convert dimensions to pixels
    cell_centers_x = cell_centers_x / um_pixel_ratio
    cell_centers_y = cell_centers_y / um_pixel_ratio
    cell_lengths = cell_lengths / um_pixel_ratio
    cell_radii = cell_radii / um_pixel_ratio

    return cell_centers_x, cell_centers_y, cell_lengths, cell_radii, cell_orientations


def get_image_dimensions(cell_centers_x, cell_centers_y):
    """
    Obtains dimensions for an image in pixels based on colony size

    @param  cell_centers_x  list or nparray of x-coordinates for cell centroids
    @param  cell_centers_y  list or nparray of y-coordinates for cell centroids
    @return img_dimensions  (width, height)
    """
    min_x = min(cell_centers_x) - 5
    max_x = max(cell_centers_x) + 5
    min_y = min(cell_centers_y) - 5
    max_y = max(cell_centers_y) + 5
    width = abs(max_x) + abs(min_x)
    height = abs(max_y) + abs(min_y)
    img_dimensions = (round(width), round(height))

    return img_dimensions


def draw_cell(draw, center_x, center_y, length, radius, orientation, fill, outline):
    """
    Draws a cell on a black/white canvas
    Based on code of Sohaib Nadeem:
    https://github.com/ingallslab/bsim-related/blob/main/bsim_related/data_processing/image_drawing.py

    @param  draw        pillow Image
    @param  center_x    cell centroid x-coordinate
    @param  center_y    cell centroid y-coordinate
    @param  length      cell length (does not inclue ends of cell)
    @param  radius      cell radius
    @param  orientation cell angle in radians
    @param  fill        intensity of the cell fill (ranges from 0 to 255 for bw image)
    @param  outline     intensity of the cell outline (ranges from 0 to 255 for bw image)
    """

    # Adjust orientations to image coordinate system
    img_orientation = -orientation  # flip y-coordinates because (0,0) is in top left corner
    img_orientation_norm = img_orientation + math.pi / 2

    # Calculate lengths
    half_length_along_axis_x = length / 2 * math.cos(img_orientation)
    half_length_along_axis_y = length / 2 * math.sin(img_orientation)
    radius_perpendicular_to_axis_x = radius * math.cos(img_orientation_norm)
    radius_perpendicular_to_axis_y = radius * math.sin(img_orientation_norm)

    # Draw rectangle
    p1 = (center_x + half_length_along_axis_x + radius_perpendicular_to_axis_x,
          center_y + half_length_along_axis_y + radius_perpendicular_to_axis_y)
    p2 = (center_x + half_length_along_axis_x - radius_perpendicular_to_axis_x,
          center_y + half_length_along_axis_y - radius_perpendicular_to_axis_y)
    p3 = (center_x - half_length_along_axis_x - radius_perpendicular_to_axis_x,
          center_y - half_length_along_axis_y - radius_perpendicular_to_axis_y)
    p4 = (center_x - half_length_along_axis_x + radius_perpendicular_to_axis_x,
          center_y - half_length_along_axis_y + radius_perpendicular_to_axis_y)
    draw.polygon([p1, p2, p3, p4], fill=fill, outline=outline)

    # Draw ends of cell
    p5 = (center_x + half_length_along_axis_x - radius, center_y + half_length_along_axis_y - radius)
    p6 = (center_x + half_length_along_axis_x + radius, center_y + half_length_along_axis_y + radius)
    p7 = (center_x - half_length_along_axis_x - radius, center_y - half_length_along_axis_y - radius)
    p8 = (center_x - half_length_along_axis_x + radius, center_y - half_length_along_axis_y + radius)
    end_1 = img_orientation_norm * 180 / math.pi
    start_1 = end_1 - 180
    draw.pieslice([p5, p6], start=start_1, end=end_1, fill=fill, outline=outline)  # start and end angle in degrees
    draw.pieslice([p7, p8], start=end_1, end=start_1, fill=fill, outline=outline)


def draw_image_bw(img_dimensions, cell_centers_x, cell_centers_y, cell_lengths, cell_radii, cell_orientations):
    """
    Draw a black and white image from cell data (cells are white, background is black)

    Any cell dimension and position must be given in pixels

    @param  img_dimensions      dimensions of the world in pixels (width, height)
    @param  cell_centers_x      list or nparray of x-coordinates for cell centroids
    @param  cell_centers_y      list or nparray of y-coordinates for cell centroids
    @param  cell_lengths        list or nparray of cell lengths
    @param  cell_radii          list or nparray of cell radii
    @param  cell_orientations   list or nparray of cell orientations
    @return img                 black and white image of cells
    """
    cell_count = len(cell_centers_x)
    img = Image.new(mode='L', size=img_dimensions, color=0)
    draw = ImageDraw.Draw(img)
    fill = 255
    outline = None

    for i in range(cell_count):
        draw_cell(draw, cell_centers_x[i], cell_centers_y[i], cell_lengths[i], cell_radii[i], cell_orientations[i],
                  fill, outline)

    return img


def calculate_colony_density(img, fig_export_path=''):
    """
    Based on code of J. Chalaturnik and S. Nadeem:
    https://github.com/ingallslab/bsim-related/blob/main/bsim_related/data_processing/img_processing.py
    Calculates the fraction of the colony occupied by cells.
    @param  img             black/white pillow image of a colony
    @param  fig_export_path directory to export images to
    @return densityCalculation         colony densityCalculation parameter (1 = colony is completely filled in)
    """
    cell_area = np.count_nonzero(np.array(img) == 255)

    # Perform morphological closing to get rid of any gaps
    img_close = skimage.morphology.closing(img, footprint=np.ones((7, 7), dtype='uint8'))

    # Get contours
    contours, hierarchy = cv2.findContours(np.copy(img_close), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Fill contours so that the image is a filled envelope of the microcolony
    img_close = cv2.drawContours(img_close, contours, -1, 255, cv2.FILLED)

    # Calculate filled area of colony
    img_props = pd.DataFrame(skimage.measure.regionprops_table(img_close, properties=["area"]))
    filled_contour_area = img_props["area"][0]

    # Calculate fraction of area occupied by cells
    density = cell_area / filled_contour_area

    # Optional: export figures of colonies to fig_export_path
    if fig_export_path:
        img_close_fromarray = Image.fromarray(img_close)
        img_close_fromarray.save(fig_export_path + 'closed_colony.png')
        img.save(fig_export_path + 'colony.png')
        print('Colony images exported to: ', fig_export_path)

    return density


def calc_density(cells, fig_export_path=''):
    """
    The main function for calculating colony density

    @param  cells           cellStates dict
    @param  fig_export_path directory to export images to
    @return densityCalculation         colony densityCalculation parameter (1 = colony is completely filled in)
    """
    # Obtain cell data in format suitable for creating black/white images
    cell_centers_x, cell_centers_y, cell_lengths, cell_radii, cell_orientations = \
        get_cell_data_to_draw_image_bw(cells, um_pixel_ratio=0.144)

    # Create image dimensions
    img_dimensions = get_image_dimensions(cell_centers_x, cell_centers_y)

    # Draw black and white image
    bw_img = draw_image_bw(img_dimensions, cell_centers_x, cell_centers_y, cell_lengths, cell_radii, cell_orientations)

    # Define export path and compute colony densityCalculation
    density = calculate_colony_density(bw_img, fig_export_path=fig_export_path)

    return density
