from collections import OrderedDict
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import interpolation


class PictureFrame(object):
    """
    Lightweight wrapper for a dictionary of NumPy arrays with shape
    constraints and additional convenience functions for joint
    indexing and resizing.  Contains named data arrays which
    must be of at least `fixed_dim` dimensions and whose first
    `fixed_dim` dimensions must match.

    Parameters
    ----------
    data : dict, optional
        Input data
    fixed_dim : int, optional
        Number of fixed dimensions shared across all arrays. Default 2.

    Examples
    --------
    >>> data = {'image': np.random.random_sample((640, 480, 3)),
                'mask' = np.random.random_sample((640, 480)) > 0.5}
    >>> pf = PictureFrame(data=data)
    >>> pf2 = pf[:10, 20:30]
    """

    def __init__(self, data=None, fixed_dim=2):

        self._data_shape = None
        self._data = {}
        self._fixed_dim = fixed_dim

        if isinstance(data, dict):
            # all members must have the same width and height
            shapes = {d.shape[:fixed_dim] for d in data.values()}

            if len(shapes) > 1:
                raise ValueError('Data shapes do not match')

            self._data_shape = shapes.pop()
            self._data = OrderedDict(data)

        elif data is not None:
            raise ValueError('Data format not understood')

    def __getattr__(self, name):
        """
        Allow for access of named data arrays as attributes
        if the attribute does not exist otherwise.
        """
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            E, V, T = sys.exc_info()
            try:
                return self._data[name]
            except KeyError:
                pass
            raise E(V)

    def __getitem__(self, args):
        """
        Either access named data array or return a new PictureFrame
        where each data array is indexed with the arguments.
        The new PictureFrame may have a different shape and number of
        fixed dimensions, depending on the indexing operation.
        """

        if isinstance(args, str):
            return self._data[args]

        else:
            data = OrderedDict()
            fixed_dim = None

            for k, v in self._data.items():
                subarray = v.__getitem__(args)
                data[k] = subarray
                # after indexing the first array, work out the number
                # of fixed dimensions for the new PictureFrame
                if fixed_dim is None:
                    fixed_dim = self._fixed_dim + subarray.ndim - v.ndim

            return PictureFrame(data, fixed_dim=fixed_dim)

    def __setitem__(self, args, values):
        """
        Either set named data array or set the values of each data array
        with the supplied PictureFrame
        """
        if isinstance(args, str):
            return self.add_array(args, values)

        elif isinstance(values, PictureFrame):
            for k, v in self._data.items():
                v[args] = values[k]
        else:
            raise ValueError('Value type not understood')

    def add_array(self, name, value):

        if self._data_shape is None:
            self._fixed_dim = 2  # TODO
            self._data_shape = value.shape[:self._fixed_dim]
        elif value.shape[:self._fixed_dim] != self._data_shape:
            raise ValueError('Data shape does not match')
        self._data[name] = value

    def __repr__(self):

        lines = []
        lines.append('PictureFrame. shape {}'.format(self._data_shape))
        lines.append('-' * len(lines[0]))
        for k, v in self._data.items():
            if v.size <= 3:
                lines.append('{} {}'.format(k, v))
            else:
                lines.append('{} {} {}'.format(k, v.shape, v.dtype))
        return '\n'.join(lines)

    def zoom(self, zoom, orders=None):
        """
        Rescale all data arrays along constrained axes
        Default to bilinear interpolation for float arrays
        and nearest neighbour for int arrays

        Parameters
        ----------
        zoom : float
            Scale factor.
        orders : dict, optional
            Override default order of interpolation for member arrays.
            Format is {'name': order}, where order is an integer.

        Returns
        -------
        scaled : PictureFrame
        """

        # zoom only base dimensions
        zoom_base = (zoom,) * self._fixed_dim

        if orders is None:
            orders = {}

        data = OrderedDict()

        for k, v in self._data.items():

            if k in orders:
                order = orders[k]
            elif issubclass(v.dtype.type, np.integer):
                order = 0
            elif v.dtype == np.bool:
                order = 0
            else:
                order = 2

            # maintain scale for remaining dimensions
            zoom = zoom_base + (1,) * (v.ndim - self._fixed_dim)
            data[k] = interpolation.zoom(v, order=order, zoom=zoom)

        return PictureFrame(data, fixed_dim=self._fixed_dim)

    def groupby(self, by):
        """
        Group PictureFrame by data array name or by array of labels.

        Parameters
        ----------
        by : string or ndarray
            Name of data array or array of label values.

        Returns
        -------
        it : generator
            Yields tuples of (label, PictureFrame) at corresponding locations
        """

        # can either use a data array by name or use a supplied label array
        if isinstance(by, str):
            labels = self.__getitem__(by)
        else:
            if by.ndim != self._fixed_dim:
                raise ValueError('Invalid dimensions for label array')
            if by.shape != self._data_shape:
                raise ValueError('Invalid shape for label array')
            labels = by

        for label in np.unique(labels):
            yield label, self.__getitem__(labels == label)

    def browse(self):
        """
        Debug method for interactive viewing of PictureFrame data.
        """
        if not self._fixed_dim == 2:
            raise NotImplementedError('Browsing only supported for image data')
        plt.ion()
        for k, v in self._data.items():
            plt.clf()
            if v.ndim == 2:
                plt.imshow(v, cmap='Greys_r')
            else:
                plt.imshow(v)
            plt.title(k)
            plt.waitforbuttonpress()
        plt.close()
        plt.ioff()
