# Copyright 2025 Yongchao Wu
# This file is part of the TorchNEP project.
# TorchNEP is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# TorchNEP is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with TorchNEP.  If not, see <http://www.gnu.org/licenses/>.

"""Current c516 input flags and named-block model parsing.

The c516 input parser supports one to seven l_max values; model files use
three to seven values, with q_222 encoded as 0 or 2.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from torchnep.model import NEPModel
from torchnep.nep import NEPCalculator


def _minimal_config(l_max_list):
    return {
        "num_types": 1, "type_names": ["X"],
        "cutoff_radial": 6.0, "cutoff_angular": 5.0,
        "n_max_radial": 4, "n_max_angular": 4,
        "basis_size_radial": 8, "basis_size_angular": 8,
        "l_max": l_max_list, "neuron": 10,
    }


@pytest.mark.parametrize("l_max,expected", [
    ([4, 2, 1], (4, 1, 1, 0)),
    ([4, 2, 0], (4, 1, 0, 0)),
    ([4, 0, 0], (4, 0, 0, 0)),
    # 4-field form: full GPUMD-core flags
    ([4, 1, 1, 1],    (4, 1, 1, 1)),
    ([4, 1, 0, 1],    (4, 1, 0, 1)),
    ([4, 1, 0, 0],    (4, 1, 0, 0)),
    # c516 permits an explicit rank with omitted higher-body flags.
    ([4],             (4, 0, 0, 0)),
])
def test_l_max_field_normalisation(l_max, expected):
    """Trailing fields normalise to 0/1 regardless of input width."""
    m = NEPModel(_minimal_config(l_max))
    got = (m.l_max_3b, m.has_q_222, m.has_q_1111, m.has_q_112)
    assert got == expected


def test_c516_named_block_fixture_loads():
    """The frozen ordinary potential uses only the c516 named-block format."""
    nep_txt = Path(__file__).resolve().parent / "data" / "nep_CrCoNi.txt"
    calc = NEPCalculator(str(nep_txt))
    assert calc.l_max_3b == 4
    assert calc.has_q_222 == 1
    assert calc.has_q_1111 == 1
    assert calc.has_q_112 == 0
    # Sanity on derived dim:
    # radial (n_max_r+1=9) + 3-body (9*4=36) + q_222 (9) + q_1111 (9) = 63
    assert calc.dim == 63
