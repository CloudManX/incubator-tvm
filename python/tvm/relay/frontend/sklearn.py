# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=invalid-name, import-self, len-as-condition, unused-argument, too-many-lines
# pylint: disable=import-outside-toplevel
"""ONNX: Open Neural Network Exchange frontend for Relay."""
import numpy as np
import tvm
from tvm.ir import IRModule

from ... import nd as _nd
from .. import analysis
from .. import expr as _expr
from .. import function as _function
from .. import op as _op
from .. import vision as _vision

from ..function import Function
from ..expr import Call, Let
from ..expr import If, Tuple, TupleGetItem
from ..expr import RefCreate, RefRead, RefWrite
from ..expr_functor import ExprFunctor
from ..adt import Match, Clause

from .common import AttrCvt, Renamer, ExprTable
from .common import get_relay_op, new_var, infer_shape, infer_channels
from .common import infer_type, get_name
from .common import infer_value as _infer_value
from .common import infer_value_simulated as _infer_value_simulated

# def _imputor():
#     def _impl(inputs, input_types):


def _SimpleImputer(op, inexpr, dshape, dtype):
    ret = _op.add(inexpr, _expr.const(1, dtype=dtype))
    # Boolean mask for nan values
    boolean_mask = _op.isnan(inexpr)

    # Zero out nan values
    nan_zeroed = _op.where(boolean_mask, _op.zeros(shape=dshape, dtype=dtype), inexpr)

    sum_col = _op.sum(nan_zeroed, axis=0)


    # if op.strategy == "mean": 

    # Convert one mask to numeric mask
    one_mask = _op.where(boolean_mask,
                        _op.zeros(shape=dshape, dtype=dtype),
                        _op.ones(shape=dshape, dtype=dtype))
    num_of_non_nan_values = _op.sum(one_mask, axis=0)
    # Vector of values to be filled to missing value for each column
    fill_col = sum_col / num_of_non_nan_values

    # elif op.strategy == "median":
    #     # full_max_val = _op.const(np.full(dshape, 100, dtype="float32"))
    #     full_max_val = _op.full_like(inexpr, _expr.const(1))
    #     # full_max_val = _op.full(_expr.const(100), tuple(dshape), "float32")

    #     data_nan_to_max = _op.where(boolean_mask, full_max_val, inexpr)
    #     sorted_col = _op.argsort(data=inexpr, axis=0)
    #     return sorted_col   

    reps = [1 if i > 0 else dshape[0] for i in range(len(dshape))]

    fill_val = _op.tile(fill_col, reps=reps)
    
    ret = _op.where(boolean_mask,
                      fill_val,
                      inexpr)
    
    return ret

def _RobustImputer(op, inexpr, dshape, dtype):
    ret = _op.add(inexpr, _expr.const(1, dtype=dtype))
    # Boolean mask for nan values
    boolean_mask = _op.isnan(inexpr)
    
    # Zero out nan values
    nan_zeroed = _op.where(boolean_mask, _op.zeros(shape=dshape, dtype=dtype), inexpr)

    sum_col = _op.sum(nan_zeroed, axis=0)

    if op.strategy == "mean":
        # Convert one mask to numeric mask
        one_mask = _op.where(boolean_mask,
                            _op.zeros(shape=dshape, dtype=dtype),
                            _op.ones(shape=dshape, dtype=dtype))
        num_of_non_nan_values = _op.sum(one_mask, axis=0)
        avg_col = sum_col / num_of_non_nan_values
        reps = [1 if i > 0 else dshape[0] for i in range(len(dshape))]
        fill_val = _op.tile(avg_col, reps=reps)
    elif op.strategy == "constant":
        if type(op.fill_values) == list:
            assert(len(op.fill_values) == dshape[-1])
            fill_val = _op.const(np.array(op.fill_values, dtype=dtype))
            fill_val = _op.tile(fill_val, reps=(4,1))
        else:
            fill_val = _op.full_like(inexpr, _op.const(op.fill_values))

    ret = _op.where(boolean_mask,
                      fill_val,
                      inexpr)
    
    return ret

def _OneHotEncoder(op, inexpr, dshape, dtype):
    cols = _op.split(inexpr, dshape[1], axis=1)

    out = [] 
    for i in range(dshape[1]):
        category = op.categories_[i]
        cat_tensor = _op.const(np.array(category, dtype=dtype))
        tiled_col = _op.tile(cols[i], (1, len(category)))
        one_hot = _op.equal(tiled_col, cat_tensor)
        one_hot_shape = [dshape[0], len(category)]
        one_hot = _op.where(one_hot,
                            _op.ones(shape=one_hot_shape, dtype=dtype),
                            _op.zeros(shape=one_hot_shape, dtype=dtype))
        out.append(one_hot)
    ret = _op.concatenate(out, axis=1) 
    return ret 

def _OneHotEncoder(op, inexpr, dshape, dtype):
    cols = _op.split(inexpr, dshape[1], axis=1)

    out = [] 
    for i in range(dshape[1]):
        category = op.categories_[i]
        cat_tensor = _op.const(np.array(category, dtype=dtype))
        tiled_col = _op.tile(cols[i], (1, len(category)))
        one_hot = _op.equal(tiled_col, cat_tensor)
        one_hot_shape = [dshape[0], len(category)]
        one_hot = _op.where(one_hot,
                            _op.ones(shape=one_hot_shape, dtype=dtype),
                            _op.zeros(shape=one_hot_shape, dtype=dtype))
        out.append(one_hot)
    ret = _op.concatenate(out, axis=1) 
    return ret 

def _ThresholdOneHotEncoder(op, inexpr, dshape, dtype):
    cols = _op.split(inexpr, dshape[1], axis=1)

    out = [] 
    for i in range(dshape[1]):
        category = op.categories[i]
        cat_tensor = _op.const(np.array(category, dtype=dtype))
        tiled_col = _op.tile(cols[i], (1, len(category)))
        one_hot = _op.equal(tiled_col, cat_tensor)
        one_hot_shape = [dshape[0], len(category)]
        one_hot = _op.where(one_hot,
                            _op.ones(shape=one_hot_shape, dtype=dtype),
                            _op.zeros(shape=one_hot_shape, dtype=dtype))
        out.append(one_hot)

    one_hot_concat = _op.concatenate(out, axis=1) 
    col_cnt = _op.sum(one_hot_concat, axis=0)
    
    threshold_mask = _op.greater(col_cnt, _op.const(op.threshold, dtype=dtype))
    threshold_num_mask = _op.where(threshold_mask,
                                   _op.ones(shape=(8,), dtype=dtype),
                                   _op.zeros(shape=(8,), dtype=dtype))
    ret = _op.bitwise_and(one_hot_concat, threshold_num_mask)
    return ret

_convert_map = {
    'SimpleImputer': _SimpleImputer,
    'RobustImputer': _RobustImputer,
    'OneHotEncoder': _OneHotEncoder,
    'ThresholdOneHotEncoder': _ThresholdOneHotEncoder
}

def sklearn_op_to_relay(op, inexpr, dshape, dtype):
    classname = type(op).__name__
    return _convert_map[classname](op, inexpr, dshape, dtype)

def from_sklearn(model,
              shape=None,
              dtype="float32"):
    print('running sklearn frontend......................................')

    try:
        import sklearn
    except ImportError:
        pass

    inexpr = _expr.var('input', shape=shape, dtype=dtype)
    outexpr = sklearn_op_to_relay(model, inexpr, shape, dtype)

    func = _function.Function(analysis.free_vars(outexpr), outexpr)

    return IRModule.from_expr(func), []

def from_auto_ml(model,
              shape=None,
              dtype="float32"):
    print('running sklearn frontend......................................')

    try:
        import sklearn
    except ImportError:
        pass

    inexpr = _expr.var('input', shape=shape, dtype=dtype)
    outexpr = sklearn_op_to_relay(model, inexpr, shape, dtype)

    func = _function.Function(analysis.free_vars(outexpr), outexpr)

    return IRModule.from_expr(func), []


    
    