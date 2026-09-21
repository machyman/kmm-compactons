import pytest, numpy as np
from kmm_compactons.compacton import universal_formulas
from kmm_compactons.conservation import evaluate_conservation_laws, verify_trig_orthogonality

class TestConservationLaws:
    @pytest.mark.parametrize("m", [2, 3, 4, 5])
    def test_trig_orthogonality(self, m):
        result = verify_trig_orthogonality(m, A=1.0, N=16384)
        assert abs(result['I2']) < 1e-5
        assert abs(result['I3']) < 1e-5
        assert abs(result['ell_minus_alpha'] - 2.0) < 1e-12
