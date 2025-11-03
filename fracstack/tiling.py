import math
from typing import List, Sequence, Tuple, Optional

import numpy as np


def split_image_into_tiles(
    image: np.ndarray,
    tile_height: int,
    tile_width: int,
    *,
    fill_value: int = 0,
) -> Tuple[List[np.ndarray], int, int, Tuple[int, int]]:
    """
    Split a 2D image into equally sized tiles, padding as needed.

    Parameters
    ----------
    image : np.ndarray
        Input image data. Must be 2D or 3D (H x W [x C]).
    tile_height : int
        Height of each tile in pixels (> 0).
    tile_width : int
        Width of each tile in pixels (> 0).
    fill_value : int, optional
        Value used to pad the image if its dimensions are not divisible by the
        tile size. Defaults to 0.

    Returns
    -------
    tiles : list of np.ndarray
        List containing the extracted tiles in row-major order.
    rows : int
        Number of tile rows.
    cols : int
        Number of tile columns.
    original_shape : tuple of int
        Tuple ``(height, width)`` describing the original image shape.
    """
    if tile_height <= 0 or tile_width <= 0:
        raise ValueError("Tile height and width must be positive integers.")

    array = np.asarray(image)
    if array.ndim < 2:
        raise ValueError("Input image must have at least two dimensions.")

    height, width = array.shape[:2]
    rows = math.ceil(height / tile_height)
    cols = math.ceil(width / tile_width)

    padded_height = rows * tile_height
    padded_width = cols * tile_width

    padded_shape = (padded_height, padded_width) + array.shape[2:]
    padded = np.full(padded_shape, fill_value, dtype=array.dtype)
    padded[:height, :width, ...] = array

    tiles: List[np.ndarray] = []
    for r in range(rows):
        y0 = r * tile_height
        y1 = y0 + tile_height
        for c in range(cols):
            x0 = c * tile_width
            x1 = x0 + tile_width
            tiles.append(padded[y0:y1, x0:x1, ...].copy())

    return tiles, rows, cols, (height, width)


def assemble_tiles(
    tiles: Sequence[np.ndarray],
    rows: int,
    cols: int,
) -> np.ndarray:
    """
    Assemble tiles back into a single image.

    Parameters
    ----------
    tiles : sequence of np.ndarray
        Tiles to assemble. Sequence length must equal ``rows * cols``.
    rows : int
        Number of tile rows.
    cols : int
        Number of tile columns.

    Returns
    -------
    np.ndarray
        Reconstructed image containing the tiles in row-major order.
    """
    if rows <= 0 or cols <= 0:
        raise ValueError("Rows and columns must be positive integers.")

    expected = rows * cols
    if len(tiles) != expected:
        raise ValueError(f"Expected {expected} tiles, got {len(tiles)}.")

    if not tiles:
        raise ValueError("Tile sequence is empty.")

    tile_shape = tiles[0].shape
    for tile in tiles:
        if tile.shape != tile_shape:
            raise ValueError("All tiles must share the same shape.")

    tile_height, tile_width = tile_shape[:2]
    out_shape = (rows * tile_height, cols * tile_width) + tile_shape[2:]
    assembled = np.empty(out_shape, dtype=tiles[0].dtype)

    index = 0
    for r in range(rows):
        y0 = r * tile_height
        y1 = y0 + tile_height
        for c in range(cols):
            x0 = c * tile_width
            x1 = x0 + tile_width
            assembled[y0:y1, x0:x1, ...] = tiles[index]
            index += 1

    return assembled


def generate_random_rearrangements(
    image: np.ndarray,
    tile_height: int,
    tile_width: int,
    *,
    count: int,
    fill_value: int = 0,
    seed: Optional[int] = None,
    max_attempts: int = 20,
) -> Tuple[List[np.ndarray], dict, List[Tuple[int, ...]]]:
    """
    Generate random rearrangements of an image by shuffling tiles.

    Parameters
    ----------
    image : np.ndarray
        Input image array (2D or 3D) to shuffle.
    tile_height : int
        Height of each tile in pixels.
    tile_width : int
        Width of each tile in pixels.
    count : int
        Number of random rearrangements to produce.
    fill_value : int, optional
        Fill value for padding. Defaults to 0.
    seed : int, optional
        Seed for the underlying RNG to produce deterministic shuffles.
    max_attempts : int, optional
        Maximum number of attempts to find a new permutation per rearrangement.

    Returns
    -------
    rearrangements : list of np.ndarray
        List containing ``count`` arrays with shuffled tiles.
    metadata : dict
        Metadata describing the tiling configuration.
    permutations : list of tuple of int
        The tile index permutations applied to create each rearrangement.
    """
    if count <= 0:
        raise ValueError("Rearrangement count must be a positive integer.")

    tiles, rows, cols, original_shape = split_image_into_tiles(
        image,
        tile_height,
        tile_width,
        fill_value=fill_value,
    )
    if not tiles:
        raise ValueError("Unable to produce tiles from the supplied image.")

    rng = np.random.default_rng(seed)
    tile_count = len(tiles)
    base_order = tuple(range(tile_count))
    seen_orders = {base_order}

    results: List[np.ndarray] = []
    permutations: List[Tuple[int, ...]] = []
    for _ in range(count):
        attempt = 0
        new_order: Optional[Tuple[int, ...]] = None
        while attempt < max_attempts:
            permutation = tuple(rng.permutation(tile_count))
            if permutation not in seen_orders:
                new_order = permutation
                break
            attempt += 1

        if new_order is None:
            # Fall back to the latest permutation even if repeated to avoid stalling
            new_order = tuple(rng.permutation(tile_count))

        seen_orders.add(new_order)
        permutations.append(new_order)
        shuffled_tiles = [tiles[index] for index in new_order]
        assembled = assemble_tiles(shuffled_tiles, rows, cols)
        height, width = original_shape
        cropped = assembled[:height, :width, ...].copy()
        results.append(cropped)

    metadata = {
        "rows": rows,
        "cols": cols,
        "tile_height": tile_height,
        "tile_width": tile_width,
        "original_height": original_shape[0],
        "original_width": original_shape[1],
        "tile_count": tile_count,
    }
    return results, metadata, permutations
