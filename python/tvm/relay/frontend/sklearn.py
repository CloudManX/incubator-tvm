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


def _SimpleImputer(op, inexpr, dshape):
    ret = _op.add(inexpr, _expr.const(1, dtype="float32"))
    # Boolean mask for nan values
    boolean_mask = _op.isnan(inexpr)

    # Zero out nan values
    nan_zeroed = _op.where(boolean_mask, _op.zeros(shape=dshape, dtype="float32"), inexpr)

    sum_col = _op.sum(nan_zeroed, axis=0)

    # Convert one mask to numeric mask
    one_mask = _op.where(boolean_mask,
                        _op.zeros(shape=dshape, dtype="float32"),
                        _op.ones(shape=dshape, dtype="float32"))

    num_of_non_nan_values = _op.sum(one_mask, axis=0)

    avg_col = sum_col / num_of_non_nan_values

    reps = [1 if i > 0 else dshape[0] for i in range(len(dshape))]

    avg_val = _op.tile(avg_col, reps=reps)
    
    ret = _op.where(boolean_mask,
                      avg_val,
                      inexpr)
    
    return ret

def _RobustImputer(op, inexpr, dshape):
    print('woshinibaba')

_convert_map = {
    'SimpleImputer': _SimpleImputer,
    'RobustImputer': _RobustImputer
}

def sklearn_op_to_relay(op, inexpr, dshape):
    classname = type(op).__name__
    return _convert_map[classname](op, inexpr, dshape)

def from_sklearn(model,
              shape=None,
              dtype="float32"):
    print('running sklearn frontend......................................')

    try:
        import sklearn
    except ImportError:
        pass

    inexpr = _expr.var('input', shape=shape, dtype=dtype)
    outexpr = sklearn_op_to_relay(model, inexpr, shape)

    func = _function.Function(analysis.free_vars(outexpr), outexpr)

    return IRModule.from_expr(func), []




    
    