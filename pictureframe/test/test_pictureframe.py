import numpy as np
from skimage.io import imread
from skimage import img_as_float, img_as_ubyte

from numpy.testing import assert_raises
from numpy.testing import assert_array_equal

from pictureframe import PictureFrame


def get_2d_data():
    shape = (100, 200)
    pf = PictureFrame({'image': np.random.random_sample(shape + (3,)),
                       'labels': np.random.random_integers(0, 10, size=shape)})
    return pf


def test_add_array():
    # test adding array
    # test attribute exists and is equal to the original data
    pf = get_2d_data()
    data = np.random.random_sample(pf._data_shape)
    pf['depth'] = data
    assert_array_equal(data, pf.depth)
    assert_array_equal(data, pf['depth'])


def test_setitem():
    # test setting all values in a pictureframe slice
    pf = get_2d_data()
    indices = np.arange(10)
    pf2 = pf[indices]

    pf2['image'] = pf2.image + 2.0
    pf2['labels'] = pf2.labels + 1

    pf[indices] = pf2
    assert_array_equal(pf[indices].image, pf2.image)


def test_array_mismatch_constraint():
    # test that trying to add a differently shaped array fails
    pf = get_2d_data()
    data = np.random.random_sample(pf._data_shape)

    def f(d):
        pf['depth'] = d[1:, :]

    assert_raises(ValueError, f, data)


def test_attribute_error_getattr():
    # test that __getattr__ raises a AttributeError rather than a KeyError
    # from attempting to access the data dictionary
    pf = PictureFrame()

    def f():
        pf.does_not_exist

    assert_raises(AttributeError, f)


def test_overwrite_array():
    pf = get_2d_data()
    old_data = pf.image.copy()
    new_data = np.random.random_sample(pf.image.shape)
    pf['image'] = new_data
    assert(~np.allclose(pf.image, old_data))
    assert_array_equal(pf.image, new_data)


def test_slicing():
    # test that we get a view of each sliced array
    pf = get_2d_data()
    pf2 = pf[:10, :7]
    assert(pf2.image.base is pf.image)
    assert(pf2.labels.base is pf.labels)


def test_dimension_reduction():
    # test that constrained dimensions are changed correctly
    pf = get_2d_data()
    pf2 = pf[[1, 0, 3], [5, 6, 7]]
    assert(pf2._fixed_dim == 1)


def test_zoom():
    # test resize method
    # test override of interpolation
    pf = get_2d_data()
    pf2 = pf.zoom(0.5)
    pf3 = pf.zoom(0.25, orders={'image': 3})

    assert(pf._data_shape[0] / 2 == pf2._data_shape[0])
    assert(pf._data_shape[1] / 2 == pf2._data_shape[1])
    assert(pf._data_shape[0] / 4 == pf3._data_shape[0])
    assert(pf._data_shape[1] / 4 == pf3._data_shape[1])
