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

import numpy as np
import tvm
from tvm import relay
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


def _SimpleImputer(op, inexpr, dshape, dtype, columns=None):
    """
    Scikit-Learn Transformer: 
    Imputation transformer for completing missing values.
    """
    boolean_mask = _op.isnan(inexpr)
    fill_col = _op.const(np.array(op.statistics_, dtype=dtype))
    input_shape = _op.shape_of(inexpr)
    reps = _op.take(input_shape, _op.const([0]))
    reps = _op.concatenate([reps, _op.const([1])], axis=0)

    fill_val = _op.tile(fill_col, reps=reps)
    indices =_op.const(np.arange(len(op.statistics_)))
    fill_val = _op.take(fill_val, indices=indices, axis=1)

    ret = _op.where(boolean_mask,
                    fill_val,
                    inexpr)
    
    return ret

def _RobustImputer(op, inexpr, dshape, dtype, columns=None):
    """
    Sagemaker-Scikit-Learn-Extension Transformer: 
    Imputation transformer for completing missing values with multi-column support.
    """
    if columns: 
        column_indices = _op.const(columns)
        inexpr = _op.take(inexpr, indices=column_indices, axis=1)

    if op.mask_function is not None:
        inf_mask = _op.isinf(inexpr)
        nan_val = _op.full_like(inexpr, _op.const(np.array(np.nan, dtype=dtype)))
        inexpr = _op.where(inf_mask, nan_val, inexpr) 
    ret = _SimpleImputer(op.simple_imputer_, inexpr, dshape, dtype, columns)

    return ret 
    
def _ThresholdOneHotEncoder(op, inexpr, dshape, dtype, columns=None):
    """
    Sagemaker-Scikit-Learn-Extension Transformer: 
    Encode categorical integer features as a one-hot numeric array, with optional restrictions on
    feature encoding.
    """
    if columns: 
        column_indices = _op.const(columns)
        inexpr = _op.take(inexpr, indices=column_indices, axis=1)

    num_cat = len(op.categories_)
    cols = _op.split(inexpr, num_cat, axis=1)

    out = [] 
    for i in range(num_cat):
        category = op.categories_[i]
        cat_tensor = _op.const(np.array(category, dtype=dtype))
        tiled_col = _op.tile(cols[i], (1, len(category)))
        one_hot_mask = _op.equal(tiled_col, cat_tensor)
        one_hot = _op.cast(one_hot_mask, dtype)
        out.append(one_hot)

    ret = _op.concatenate(out, axis=1) 
    return ret

def _RobustStandardScaler(op, inexpr, dshape, dtype, columns=None):
    """
    Sagemaker-Scikit-Learn-Extension Transformer: 
    Standardize features by removing the mean and scaling to unit variance
    """
    scaler = op.scaler_
    ret = _op.subtract(inexpr, _op.const(np.array(scaler.mean_, dtype), dtype))
    ret = _op.divide(ret, _op.const(np.array(scaler.scale_, dtype), dtype))
    return ret

def _ColumnTransformer(op, inexpr, dshape, dtype, columns=None):
    """
    Scikit-Learn Compose: 
    Applies transformers to columns of an array 
    """
    out = []
    for _, pipe, cols in op.transformers_:
        mod = pipe.steps[0][1]
        out.append(sklearn_op_to_relay(mod, inexpr, dshape, dtype, cols))
    
    return _op.concatenate(out, axis=1)

def _RobustOrdinalEncoder(op, inexpr, dshape, dtype, columns=None):
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

        # one_hot_mask_cols = _op.split(one_hot_mask, len(category), axis=1)
        # unseen_mask = one_hot_mask_cols[0]
        # for j in range(1, len(category)):
        #     unseen_mask = _op.logical_or(unseen_mask, one_hot_mask_cols[j])
        seen_mask = _op.cast(_op.sum(one_hot, axis=1), dtype="bool")
        extra_class = _op.full_like(ordinal, _op.const(len(category), dtype=dtype))
        robust_ordinal = _op.where(seen_mask, ordinal, extra_class)
        out.append(robust_ordinal)
        
    ret = _op.concatenate(out, axis=1) 
    return ret 

def _RobustLabelEncoder(op, inexpr, dshape, dtype, columns=None):
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

def _NALabelEncoder(op, inexpr, dshape, dtype, columns=None):
    flattened_inexpr = _op.reshape(inexpr, newshape=(-1, 1))
    # Hardcoded flattened shape to be (?, 1)
    flattened_dshape = (relay.Any(), 1)
    ri_out = _RobustImputer(op.model_, flattened_inexpr, flattened_dshape, dtype)
    ret = _op.reshape(ri_out, newshape=-1)
    return ret

def _StandardScaler(op, inexpr, dshape, dtype, columns=None):
    ret = _op.subtract(inexpr, _op.const(np.array(op.mean_, dtype), dtype))
    ret = _op.divide(ret, _op.const(np.array(op.scale_, dtype), dtype))
    return ret

def _RobustStandardScaler(op, inexpr, dshape, dtype, columns=None):
    scaler = op.scaler_
    ret = _op.subtract(inexpr, _op.const(np.array(scaler.mean_, dtype), dtype))
    ret = _op.divide(ret, _op.const(np.array(scaler.scale_, dtype), dtype))
    return ret

def _KBinsDiscretizer(op, inexpr, dshape, dtype, columns=None):
    bin_edges = np.transpose(np.vstack(op.bin_edges_))
    # for bin_edge in bin_edges:

    out = _op.full_like(inexpr, _op.const(0, dtype=dtype))

    for i in range(1, len(bin_edges)-1):
        indices_mask = _op.full_like(inexpr, _op.const(i, dtype=dtype))
        bin_edge = _op.const(bin_edges[i])
        bin_mask = _op.greater_equal(inexpr, bin_edge)
        out = _op.where(bin_mask, indices_mask, out)
    
    return out

def _TfidfVectorizer(op, inexpr, dshape, dtype, columns=None):
    if op.use_idf:
        idf = _op.const(np.array(op.idf_, dtype=dtype), dtype=dtype)
        tfidf = _op.multiply(idf, inexpr)
        if op.sublinear_tf:
            tfidf = _op.add(tfidf, _op.const(1, dtype))
        ret = _op.nn.l2_normalize(tfidf, eps=.0001, axis=[1])
    else:
        ret = _op.nn.l2_normalize(inexpr, eps=.0001, axis=[1])
    return ret

# Buggy - needs fix
def _MultiColumnTfidfVectorizer(op, inexpr, dshape, dtype, columns=None):
    out = []
    data_rows = _op.split(inexpr,dshape[0],axis=0)
    for i in range(dshape[0]):
        if op.vectorizers_[i]:
            dshape_i = _op.shape_of(data_rows[i])
            tfidf_features = _TfidfVectorizer(op.vectorizers_[i],data_rows[i],dshape_i, dtype)
            # if op.vocabulary_sizes and tfidf_features.shape[1] < op.vocabulary_sizes[i]:
            #     tfidf_features = sp.csr_matrix(
            #         (tfidf_features.data, tfidf_features.indices, tfidf_features.indptr),
            #         shape=(tfidf_features.shape[0], op.vocabulary_sizes[i]),
            #     )
            out.append(tfidf_features)
    ret = _op.stack(out, axis=1)
    return ret


def _LogExtremeValuesTransformer(op, inexpr, dshape, dtype, columns=None):
    n_features = dshape[1]
    # if n_features != op.n_input_features_:
    #         raise ValueError("X shape does not match training shape.")
    out = []
    cols = _op.split(inexpr, n_features, axis=1)
    for j in range(n_features):
        if j in op.cols_to_transform_:
            if j in op.nonnegative_cols_:
                out.append(_op.log(_op.add(cols[j], _op.const(1, dtype))))
            else:
                sign_col = _op.sign(cols[j])
                out.append(_op.multiply(sign_col,_op.log(_op.add(_op.abs(cols[j]), _op.const(1, dtype)))))
        else:
            out.append(cols[j])
    return _op.stack(out, axis=1)

def _PCA(op, inexpr, dshape, dtype, columns=None):
    eigvec = _op.const(np.array(op.components_, dtype))
    ret = _op.nn.dense(inexpr, eigvec)
    return ret

_convert_map = {
    'ColumnTransformer':_ColumnTransformer,
    'SimpleImputer': _SimpleImputer,
    'RobustImputer': _RobustImputer,
    'RobustLabelEncoder': _RobustLabelEncoder,
    'RobustOrdinalEncoder': _RobustOrdinalEncoder,
    'NALabelEncoder': _NALabelEncoder,
    'StandardScaler': _StandardScaler,
    'KBinsDiscretizer': _KBinsDiscretizer,
    'RobustStandardScaler': _RobustStandardScaler,
    'ThresholdOneHotEncoder': _ThresholdOneHotEncoder,
    'TfidfVectorizer': _TfidfVectorizer,
    'MultiColumnTfidfVectorizer': _MultiColumnTfidfVectorizer,
    'LogExtremeValuesTransformer':_LogExtremeValuesTransformer,
    'PCA': _PCA
}

def sklearn_op_to_relay(op, inexpr, dshape, dtype, columns=None):
    classname = type(op).__name__
    return _convert_map[classname](op, inexpr, dshape, dtype, columns)

def from_sklearn(model,
                 shape=None,
                 dtype="float32",
                 columns=None):
    try:
        import sklearn
    except ImportError as e:
        raise ImportError(
            "Unable to import scikit-learn which is required {}".format(e))
    inexpr = _expr.var('input', shape=shape, dtype=dtype)
    outexpr = sklearn_op_to_relay(model, inexpr, shape, dtype, columns)

    func = _function.Function(analysis.free_vars(outexpr), outexpr)
    return IRModule.from_expr(func), []

def from_auto_ml(model,
                shape=None,
                dtype="float32"):

    try:
        import sklearn
    except ImportError as e:
        raise ImportError(
            "Unable to import scikit-learn which is required {}".format(e))

    outexpr = _expr.var('input', shape=shape, dtype=dtype)
    for _, transformer in model.feature_transformer.steps:
        outexpr = sklearn_op_to_relay(transformer, outexpr, shape, dtype, None)

    func = _function.Function(analysis.free_vars(outexpr), outexpr)
    return IRModule.from_expr(func), []