# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import numpy as np
import os
import six



logger = logging.getLogger(__name__)


def assert_allclose(x, y, atol=1e-5, rtol=1e-4, verbose=True):
    """Asserts if some corresponding element of x and y differs too much.

    This function can handle both CPU and GPU arrays simultaneously.

    Args:
        x: Left-hand-side array.
        y: Right-hand-side array.
        atol (float): Absolute tolerance.
        rtol (float): Relative tolerance.
        verbose (bool): If ``True``, it outputs verbose messages on error.

    """
    try:
        #logging.warning("int8_outputis {} and fp32 output is {} ".format(x, y))
        np.testing.assert_allclose(
            x, y, atol=atol, rtol=rtol, verbose=verbose)
        return True
    except AssertionError as e:
        f = six.StringIO()
        f.write(str(e) + '\n\n')
        f.write(
            'assert_allclose failed: \n' +
            '  shape: {} {}\n'.format(x.shape, y.shape) +
            '  dtype: {} {}\n'.format(x.dtype, y.dtype))
        if x.shape == y.shape:
            xx = x if x.ndim != 0 else x.reshape((1,))
            yy = y if y.ndim != 0 else y.reshape((1,))
            err = np.abs(xx - yy)
            i = np.unravel_index(np.argmax(err), err.shape)
            f.write(
                '  i: {}\n'.format(i) +
                '  x[i]: {}\n'.format(xx[i]) +
                '  y[i]: {}\n'.format(yy[i]) +
                '  err[i]: {}\n'.format(err[i]))
        opts = np.get_printoptions()
        try:
            np.set_printoptions(threshold=10000)
            f.write('x: ' + np.array2string(x, prefix='x: ') + '\n')
            f.write('y: ' + np.array2string(y, prefix='y: ') + '\n')
        finally:
            np.set_printoptions(**opts)
            #raise AssertionError(f.getvalue())
            logging.warning(f.getvalue())
            return False





def assert_compare(x, y, atol=1e-5, method='ALL'):
    """method can be MSE, MAE and RMSE"""
    mae=0
    mse=0
    rmse=0
    result=0
    if method=='MAE':
        mae = np.abs(x-y).mean()
        result=mae
    elif method=='RMSE':
        rmse=np.sqrt(np.square(x - y).mean())
        result=rmse
        #result=np.sqrt(((x - y) ** 2).mean())
    elif method=='MSE':
        mse = np.square(x - y).mean()
        result=mse
        #result=((x - y) ** 2).mean()
    else:
        mae = np.abs(x-y).mean()
        rmse=np.sqrt(np.square(x - y).mean())
        mse = np.square(x - y).mean()

    if result > atol or (method =='ALL' and (mae > atol or rmse > atol or mse > atol) ):
        f = six.StringIO()
        f.write(
            'assert_compare failed: \n' +
            '  atol: {} \n'.format(atol) +
            '  method: {}\n'.format(method) +
            '  MAE: {}\n'.format(mae) +
            '  MSE: {}\n'.format(mse) +
            '  RMSE: {}\n'.format(rmse) +
            '  shape: {} {}\n'.format(x.shape, y.shape) +
            '  dtype: {} {}\n'.format(x.dtype, y.dtype))
        if x.shape == y.shape:
            xx = x if x.ndim != 0 else x.reshape((1,))
            yy = y if y.ndim != 0 else y.reshape((1,))
            err = np.abs(xx - yy)
            i = np.unravel_index(np.argmax(err), err.shape)
            f.write(
                '  i: {}\n'.format(i) +
                '  x[i]: {}\n'.format(xx[i]) +
                '  y[i]: {}\n'.format(yy[i]) +
                '  err[i]: {}\n'.format(err[i]))
        opts = np.get_printoptions()
        try:
            np.set_printoptions(threshold=10000)
            f.write('x: ' + np.array2string(x, prefix='x: ') + '\n')
            f.write('y: ' + np.array2string(y, prefix='y: ') + '\n')
        finally:
            np.set_printoptions(**opts)
            logging.warning(f.getvalue())
            return False
    else:
        return True

