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
    boolean_mask = _op.isnan(inexpr)

    fill_col = _op.const(np.array(op.statistics_, dtype=dtype))
    reps = [1 if i > 0 else dshape[0] for i in range(len(dshape))]

    fill_val = _op.tile(fill_col, reps=reps)

    ret = _op.where(boolean_mask,
                    fill_val,
                    inexpr)
    
    return ret

def _RobustImputer(op, inexpr, dshape, dtype):
    if not op.mask_function:
        inf_mask = _op.isinf(inexpr)
        nan_val = _op.full_like(inexpr, _op.const(np.array(np.nan, dtype=dtype)))
        inexpr = _op.where(inf_mask, nan_val, inexpr) 
    return _SimpleImputer(op.simple_imputer_, inexpr, dshape, dtype)

def _OneHotEncoder(op, inexpr, dshape, dtype):
    cols = _op.split(inexpr, dshape[1], axis=1)

    out = [] 
    for i in range(dshape[1]):
        category = op.categories_[i]
        cat_tensor = _op.const(np.array(category, dtype=dtype))
        tiled_col = _op.tile(cols[i], (1, len(category)))
        one_hot_mask = _op.equal(tiled_col, cat_tensor)
        one_hot_shape = [dshape[0], len(category)]
        one_hot = _op.where(one_hot_mask,
                            _op.ones(shape=one_hot_shape, dtype=dtype),
                            _op.zeros(shape=one_hot_shape, dtype=dtype))
        out.append(one_hot)
    ret = _op.concatenate(out, axis=1) 
    return ret 

def _LabelEncoder(op, inexpr, dshape, dtype):
    is_inverse = False

    class_mask = []
    for i in range(len(op.classes_)):
        val = _op.const(i, dtype) if is_inverse else _op.const(np.array(op.classes_[i], dtype), dtype)
        class_mask.append(_op.equal(inexpr, val))
    return class_mask[0]

    for i in range(len(op.classes_)):
        if is_inverse:
            label_mask = _op.full_like(inexpr, _op.const(np.array(op.classes_[i], dtype), dtype=dtype))
        else:
            label_mask = _op.full_like(inexpr, _op.const(i, dtype=dtype))

        if i == 0:
            out = _op.where(class_mask[i], label_mask, inexpr)
            continue
        out = _op.where(class_mask[i], label_mask, out)

    return out

def _RobustLabelEncoder(op, inexpr, dshape, dtype):
    is_inverse = False

    class_mask = []
    for i in range(len(op.classes_)):
        val = _op.const(i, dtype) if is_inverse else _op.const(np.array(op.classes_[i], dtype), dtype)
        class_mask.append(_op.equal(inexpr, val))
    for i in range(len(op.classes_)):
        if is_inverse:
            label_mask = _op.full_like(inexpr, _op.const(np.array(op.classes_[i], dtype), dtype=dtype))
        else:
            label_mask = _op.full_like(inexpr, _op.const(i, dtype=dtype))

        if i == 0:
            out = _op.where(class_mask[i], label_mask, inexpr)
            continue
        out = _op.where(class_mask[i], label_mask, out)
                
    if op.fill_unseen_labels:
        unseen_mask = class_mask[0]
        for mask in class_mask[1:]:
            unseen_mask = _op.logical_or(unseen_mask, mask)
        unseen_mask = _op.logical_not(unseen_mask)
        unseen_label = _op.const(-1, dtype=dtype) if is_inverse else _op.const(np.array(len(op.classes_)), dtype=dtype)
        label_mask = _op.full_like(inexpr, unseen_label)
        out = _op.where(unseen_mask, label_mask, out)

    return out

def _NALabelEncoder(op, inexpr, dshape, dtype):
    flattened_inexpr = _op.reshape(inexpr, newshape=-1)
    flattened_dshape = [np.prod(dshape, dtype=np.int32)]
    ret = _RobustImputer(op.model_, flattened_inexpr, flattened_dshape, dtype)
    return ret

def _OneHotEncoder(op, inexpr, dshape, dtype):
    cols = _op.split(inexpr, dshape[1], axis=1)

    out = [] 
    for i in range(dshape[1]):
        category = op.categories_[i]
        cat_tensor = _op.const(np.array(category, dtype=dtype))
        tiled_col = _op.tile(cols[i], (1, len(category)))
        one_hot_mask = _op.equal(tiled_col, cat_tensor)
        one_hot_shape = [dshape[0], len(category)]
        one_hot = _op.where(one_hot_mask,
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

def _OrdinalEncoder(op, inexpr, dshape, dtype):
    cols = _op.split(inexpr, dshape[1], axis=1)

    out = [] 
    for i in range(dshape[1]):
        category = op.categories_[i]
        cat_tensor = _op.const(np.array(category, dtype=dtype))
        tiled_col = _op.tile(cols[i], (1, len(category)))
        one_hot_mask = _op.equal(tiled_col, cat_tensor)
        one_hot_shape = [dshape[0], len(category)]
        one_hot = _op.where(one_hot_mask,
                            _op.ones(shape=one_hot_shape, dtype=dtype),
                            _op.zeros(shape=one_hot_shape, dtype=dtype))
        offset = _op.const(np.arange(-1, len(category)-1, dtype=dtype))
        zeros = _op.full_like(one_hot, _op.const(0, dtype=dtype))
        ordinal_col =_op.where(one_hot_mask, _op.add(one_hot, offset), zeros)
        ordinal = _op.expand_dims(_op.sum(ordinal_col, axis=1), -1)
        out.append(ordinal)
    ret = _op.concatenate(out, axis=1) 
    return ret 

def _RobustOrdinalEncoder(op, inexpr, dshape, dtype):
    cols = _op.split(inexpr, dshape[1], axis=1)

    out = [] 
    for i in range(dshape[1]):
        category = op.categories_[i]
        cat_tensor = _op.const(np.array(category, dtype=dtype))
        tiled_col = _op.tile(cols[i], (1, len(category)))
        one_hot_mask = _op.equal(tiled_col, cat_tensor)
        one_hot_shape = [dshape[0], len(category)]
        one_hot = _op.where(one_hot_mask,
                            _op.ones(shape=one_hot_shape, dtype=dtype),
                            _op.zeros(shape=one_hot_shape, dtype=dtype))
        offset = _op.const(np.arange(-1, len(category)-1, dtype=dtype))
        zeros = _op.full_like(one_hot, _op.const(0, dtype=dtype))
        ordinal_col =_op.where(one_hot_mask, _op.add(one_hot, offset), zeros)
        ordinal = _op.expand_dims(_op.sum(ordinal_col, axis=1), -1)

        one_hot_mask_cols = _op.split(one_hot_mask, len(category), axis=1)
        unseen_mask = one_hot_mask_cols[0]
        for j in range(1, len(category)):
            unseen_mask = _op.logical_or(unseen_mask, one_hot_mask_cols[j])
        unseen_mask = _op.logical_not(unseen_mask)
        extra_class = _op.full_like(ordinal, _op.const(len(category), dtype=dtype))
        robust_ordinal = _op.where(unseen_mask, extra_class, ordinal)
        out.append(robust_ordinal)
        
    ret = _op.concatenate(out, axis=1) 
    return ret 

def _StandardScaler(op, inexpr, dshape, dtype):
    ret = _op.subtract(inexpr, _op.const(np.array(op.mean_, dtype), dtype))
    ret = _op.divide(ret, _op.const(np.array(op.scale_, dtype), dtype))
    return ret

def _RobustStandardScaler(op, inexpr, dshape, dtype):
    op = op.scaler_
    ret = _op.subtract(inexpr, _op.const(np.array(op.mean_, dtype), dtype))
    ret = _op.divide(ret, _op.const(np.array(op.scale_, dtype), dtype))
    return ret

def _KBinsDiscretizer(op, inexpr, dshape, dtype):
    bin_edges = np.transpose(np.vstack(op.bin_edges_))
    # for bin_edge in bin_edges:

    out = _op.full_like(inexpr, _op.const(0, dtype=dtype))

    for i in range(1, len(bin_edges)-1):
        indices_mask = _op.full_like(inexpr, _op.const(i, dtype=dtype))
        bin_edge = _op.const(bin_edges[i])
        bin_mask = _op.greater_equal(inexpr, bin_edge)
        out = _op.where(bin_mask, indices_mask, out)
    
    return out

def _TfidfVectorizer(op, inexpr, dshape, dtype):
    if op.use_idf:
        idf = _op.const(np.array(op.idf_, dtype=dtype), dtype=dtype)
        tfidf = _op.multiply(idf, inexpr)
        if op.sublinear_tf:
            tfidf = _op.add(tfidf, _op.const(1, dtype))
        ret = _op.nn.l2_normalize(tfidf, eps=.0001, axis=[1])
    else:
        ret = _op.nn.l2_normalize(inexpr, eps=.0001, axis=[1])
    
    return ret

def _PCA(op, inexpr, dshape, dtype):
    eigvec = _op.const(np.array(op.components_, dtype))
    ret = _op.nn.dense(inexpr, eigvec)
    return ret


_convert_map = {
    'SimpleImputer': _SimpleImputer,
    'RobustImputer': _RobustImputer,
    'OneHotEncoder': _OneHotEncoder,
    'LabelEncoder': _LabelEncoder,
    'RobustLabelEncoder': _RobustLabelEncoder,
    'RobustOrdinalEncoder': _RobustOrdinalEncoder,
    'NALabelEncoder': _NALabelEncoder,
    'StandardScaler': _StandardScaler,
    'KBinsDiscretizer': _KBinsDiscretizer,
    'RobustStandardScaler': _RobustStandardScaler,
    'ThresholdOneHotEncoder': _ThresholdOneHotEncoder,
    'TfidfVectorizer': _TfidfVectorizer,
    'PCA': _PCA
}

def sklearn_op_to_relay(op, inexpr, dshape, dtype):
    classname = type(op).__name__
    return _convert_map[classname](op, inexpr, dshape, dtype)

def from_sklearn(model,
              shape=None,
              dtype="float32"):
    # print('running sklearn frontend......................................')

    try:
        import sklearn
    except ImportError:
        pass

    inexpr = _expr.var('input', shape=shape, dtype=dtype)
    outexpr = sklearn_op_to_relay(model, inexpr, shape, dtype)

    func = _function.Function(analysis.free_vars(outexpr), outexpr)

    return IRModule.from_expr(func), []

# def from_auto_ml(model,
#               shape=None,
#               dtype="float32"):
#     print('running sklearn frontend......................................')

#     try:
#         import sklearn
#     except ImportError:
#         pass

#     inexpr = _expr.var('input', shape=shape, dtype=dtype)
#     outexpr = sklearn_op_to_relay(model, inexpr, shape, dtype)

#     func = _function.Function(analysis.free_vars(outexpr), outexpr)

#     return IRModule.from_expr(func), []


    
    