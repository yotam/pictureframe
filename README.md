# pictureframe

The pictureframe library contains the `PictureFrame` class, a DataFrame-like container for data of arbitrary dimensions. The main use-case is providing a clean way to store image data along with other array data such as masks, pixel labels and depth. The PictureFrame supports joint slicing and indexing of all data, along with joint resize and groupby functions. For now, please see the examples folder for a quick start guide.

Another N-dimensional DataFrame library is [xray](https://github.com/xray/xray), which provides the `DataArray` class. xray originates in the Earth sciences but it may be a great choice for vision and image processing code too. Watch this space.

## Key differences from xray DataArray

* xray is a much more fully featured project. The `DataArray` allows for more options in terms of data alignment and has support for different backends. In contrast, a `PictureFrame` is a very lightweight wrapper around a dict of NumPy ndarrays and can be used quickly with existing code.
* `DataArray` uses labeled axes and broadcasts operations according to these. `PictureFrame` shares the first `fixed_dim` elements of each array shape.

## Key differences from Pandas DataFrame

* Higher dimensional data, rather than just a 2D tabular structure.
* Arrays can have varying dimensions, only the first `fixed_dim` dimensions must match. This allows for common indexing of data such as RGB, depth, label distributions, features and so on.

## Key differences from scikit-image `ImageCollection` and `MultiImage`

* Slicing and indexing operate on the underlying array data rather than selecting a subset of the images.
* Images are constrained to match on first dimensions.
* Not constrained to image data, e.g. indexing can return a PictureFrame with fewer constrained dimensions.

# Wishlist

pictureframe is very much a work in progress. Pull requests and feature suggestions are very welcome.

This is intended to be a utility library to help write cleaner image or voxel processing code. It is meant to be "orthogonal" to existing processing functions in SciPy or Scikit-Image. So functions that help eliminate repetitive code and that are complementary to these libraries, as well as examples of how to use the PictureFrame class to get things done would be great. Here are some ideas:

- "view as windows" and "view as blocks" functions for whole PictureFrame
- groupby extension returning views by finding bounding boxes
- more constructor options
- option for generating arrays of indices over fixed dimensions
- investigate serialization
- testing and examples for 3D and higher dimensional data
- tests, documentation, suggestions for improved naming, etc
