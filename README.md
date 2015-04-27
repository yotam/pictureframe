# pictureframe

The pictureframe library contains the PictureFrame class, a DataFrame-like container for data of arbitrary dimensions. The main use-case is providing a clean way to store image data along with other array data such as masks, pixel labels and depth. The PictureFrame supports joint slicing and indexing of all data, along with joint resize and groupby functions. For now, please see the examples folder for a quick start guide.

## Wishlist

pictureframe is very much a work in progress. Pull requests and feature suggestions are very welcome.

This is intended to be a utility library to help write cleaner image or voxel processing code. It is meant to be "orthogonal" to existing processing functions in SciPy or Scikit-Image. So functions that help eliminate repetitive code and that are complementary to these libraries, as well as examples of how to use the PictureFrame class to get things done would be great. Here are some ideas:

- "view as windows" and "view as blocks" functions for whole PictureFrame
- groupby extension returning views by finding bounding boxes
- more constructor options
- option for generating arrays of indices over fixed dimensions
- investigate serialization
- testing and examples for 3D and higher dimensional data
- tests, documentation, suggestions for improved naming, etc
