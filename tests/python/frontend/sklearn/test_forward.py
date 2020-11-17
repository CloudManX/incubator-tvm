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
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sagemaker_sklearn_extension.impute import RobustImputer
from sagemaker_sklearn_extension.preprocessing import (
    RobustStandardScaler,
    ThresholdOneHotEncoder,
    RobustOrdinalEncoder,
    NALabelEncoder,
    LogExtremeValuesTransformer,
    log_transform,
)
from sagemaker_sklearn_extension.feature_extraction.text import MultiColumnTfidfVectorizer
from tvm import topi
import tvm.topi.testing
import tvm
import tvm.testing
from tvm import te
from tvm import relay
from tvm.contrib import graph_runtime

class SklearnTestHelper:
    def __init__(self, target="llvm", ctx=tvm.cpu(0)):
        self.compiled_model = None
        self.target = target
        self.ctx = ctx

    def compile(self, model, dshape, dtype, columns=None, auto_ml=False):
        if auto_ml:
            mod, _ = relay.frontend.from_auto_ml(model, dshape, dtype)
        else:
            mod, _ = relay.frontend.from_sklearn(model, dshape, dtype, columns)
        self.ex = relay.create_executor("vm", mod=mod, ctx=self.ctx, target=self.target)

    def run(self, data):
        result = self.ex.evaluate()(data)
        return result.asnumpy()


def _test_model_impl(helper, model, dshape, input_data):
    helper.compile(model, dshape, "float32")
    sklearn_out = model.transform(input_data)
    tvm_out = helper.run(input_data)
    print(type(model).__name__)
    print(sklearn_out)
    print(tvm_out)
    print("\n\n************************************\n\n")
    tvm.testing.assert_allclose(sklearn_out, tvm_out, rtol=1e-5, atol=1e-5)


def test_simple_imputer():
    st_helper = SklearnTestHelper()
    data = np.array(
        [[4, 5, np.nan, 7], [0, np.nan, 2, 3], [8, 9, 10, 11], [np.nan, 13, 14, 15]],
        dtype=np.float32,
    )

    imp_mean = SimpleImputer(missing_values=np.nan, strategy="median")
    imp_mean.fit(data)

    dshape = (relay.Any(), len(data[0]))
    _test_model_impl(st_helper, imp_mean, dshape, data)


def test_robust_imputer():
    st_helper = SklearnTestHelper()
    data = np.array(
        [[4, 5, np.nan, 7], [0, np.nan, 2, 3], [8, 9, 10, 11], [np.nan, 13, 14, 15]],
        dtype=np.float32,
    )

    ri = RobustImputer(dtype=None, strategy="constant", fill_values=np.nan, mask_function=None)
    ri.fit(data)

    dshape = (relay.Any(), len(data[0]))
    _test_model_impl(st_helper, ri, dshape, data)


def test_robust_scaler():
    st_helper = SklearnTestHelper()
    rss = RobustStandardScaler()

    data = np.array([[0, 0], [0, 0], [1, 1], [1, 1]], dtype=np.float32)
    rss.fit(data)

    dshape = (relay.Any(), len(data[0]))
    _test_model_impl(st_helper, rss, dshape, data)


def test_threshold_onehot_encoder():
    st_helper = SklearnTestHelper()
    tohe = ThresholdOneHotEncoder()

    data = np.array([[10, 1, 7], [11, 3, 8], [11, 2, 9]], dtype=np.int32)
    tohe.fit(data)
    tohe.categories_ = [[10, 11], [1, 2, 3], [7, 8, 9]]

    dshape = (relay.Any(), len(data[0]))
    st_helper.compile(tohe, dshape, "int32")
    sklearn_out = tohe.transform(data).toarray()
    tvm_out = st_helper.run(data)
    tvm.testing.assert_allclose(sklearn_out, tvm_out, rtol=1e-5, atol=1e-5)


def test_column_transfomer():
    st_helper = SklearnTestHelper()

    data = np.array(
        [[4, 5, np.nan, 7], [0, np.nan, 2, 3], [8, 9, 10, 11], [np.nan, 13, 14, 15]],
        dtype=np.float32,
    )

    pipeline = Pipeline(
        steps=[("robustimputer", RobustImputer(fill_values=np.nan, strategy="constant"))]
    )
    ct = ColumnTransformer(transformers=[("numeric_processing", pipeline, [0, 1, 2, 3])])
    ct.fit(data)

    dshape = (relay.Any(), relay.Any())
    _test_model_impl(st_helper, ct, dshape, data)


def test_robust_ordinal_encoder():
    st_helper = SklearnTestHelper()
    roe = RobustOrdinalEncoder()
    data = np.array([[0, 1], [0, 4], [1, 2], [1, 10]], dtype=np.float32)
    roe.fit(data)
    dshape = (relay.Any(), len(data[0]))
    _test_model_impl(st_helper, roe, dshape, data)


def test_na_label_encoder():
    st_helper = SklearnTestHelper()
    nle = NALabelEncoder()
    i_put = np.array([[1, 2, 2, 6]], dtype=np.float32)
    nle.fit(i_put)
    data = np.array([[np.nan, 0, 1, 2, 6]], dtype=np.float32)
    dshape = (relay.Any(), len(data))
    _test_model_impl(st_helper, nle, dshape, data)


def test_standard_scaler():
    st_helper = SklearnTestHelper()
    ss = StandardScaler()
    data = np.array([[0, 0], [0, 0], [1, 1], [1, 1]], dtype=np.float32)
    ss.fit(data)
    dshape = (relay.Any(), len(data[0]))
    _test_model_impl(st_helper, ss, dshape, data)


def test_kbins_discretizer():
    st_helper = SklearnTestHelper()
    kd = KBinsDiscretizer(n_bins=2, encode="ordinal", strategy="uniform")
    data = np.array(
        [[-2, 1, -4, -1], [-1, 2, -3, -0.5], [0, 3, -2, 0.5], [1, 4, -1, 2]], dtype=np.float32
    )
    kd.fit(data)
    dshape = (relay.Any(), len(data[0]))
    _test_model_impl(st_helper, kd, dshape, data)


def test_tfidf_vectorizer():
    st_helper = SklearnTestHelper()
    tiv = TfidfVectorizer()
    corpus = [
        "This is the first document.",
        "This document is the second document.",
        "And this is the third one.",
        "Is this the first document?",
    ]
    sklearn_out = tiv.fit_transform(corpus).toarray()
    vectorizer = CountVectorizer(dtype=np.float32)
    data = vectorizer.fit_transform(corpus).toarray()
    dshape = (relay.Any(), len(data[0]))
    st_helper.compile(tiv, dshape, "float32")
    tvm_out = st_helper.run(data)
    tvm.testing.assert_allclose(sklearn_out, tvm_out, rtol=1e-5, atol=1e-5)


# Buggy - needs fix
def test_multicolumn_tfidf_vectorizer():
    st_helper = SklearnTestHelper()
    mctiv = MultiColumnTfidfVectorizer()
    corpus = np.array(
        [
            ["Cats eat rats.", "Rats are mammals."],
            ["Dogs chase cats.", "Cats have ears."],
            ["People like dogs.", "People are mammals."],
            ["People hate rats.", "Rats are quite smart."],
        ]
    )
    mctiv.fit(corpus)
    sklearn_out = mctiv.transform(corpus)
    data = np.column_stack(
        (
            CountVectorizer(dtype=np.float32).fit_transform(corpus[:, 0]).todense(),
            CountVectorizer(dtype=np.float32).fit_transform(corpus[:, 1]).todense(),
        )
    )
    dshape = (relay.Any(), 1)
    st_helper.compile(mctiv, dshape, "float32")
    tvm_out = st_helper.run(data)
    tvm.testing.assert_allclose(sklearn_out, tvm_out, rtol=1e-5, atol=1e-5)


def test_pca():
    st_helper = SklearnTestHelper()
    pca = PCA(n_components=2)
    data = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]], dtype=np.float32)
    pca.fit(data)
    dshape = (relay.Any(), len(data[0]))
    _test_model_impl(st_helper, pca, dshape, data)


def test_log_extreme_values_transformer():
    st_helper = SklearnTestHelper()
    levt = LogExtremeValuesTransformer(threshold_std=2.0)
    x_extreme_vals = np.array(
        [
            [0.0, 0.0, 0.0],
            [-1.0, 1.0, 1.0],
            [-2.0, 2.0, 2.0],
            [-3.0, 3.0, 3.0],
            [-4.0, 4.0, 4.0],
            [-5.0, 5.0, 5.0],
            [-6.0, 6.0, 6.0],
            [-7.0, 7.0, 7.0],
            [-8.0, 8.0, 8.0],
            [-9.0, 9.0, 9.0],
            [-10.0, 10.0, 10.0],
            [-1e5, 1e6, 11.0],
        ],
        dtype=np.float32,
    )
    x_log_extreme_vals = np.column_stack(
        [
            log_transform(x_extreme_vals.copy()[:, 0]),
            log_transform(x_extreme_vals.copy()[:, 1]),
            x_extreme_vals[:, 2],
        ]
    )
    sklearn_out = levt.fit_transform(x_log_extreme_vals)
    dshape = (relay.Any(), len(x_log_extreme_vals[0]))
    st_helper.compile(levt, dshape, "float32")
    tvm_out = st_helper.run(x_log_extreme_vals)
    tvm.testing.assert_allclose(sklearn_out, tvm_out, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    test_simple_imputer()
    test_robust_imputer()
    test_robust_scaler()
    test_column_transfomer()
    test_threshold_onehot_encoder()
    test_robust_ordinal_encoder()
    test_na_label_encoder()
    test_standard_scaler()
    test_kbins_discretizer()
    test_tfidf_vectorizer()
    # test_multicolumn_tfidf_vectorizer()
    test_log_extreme_values_transformer()
    test_pca()
