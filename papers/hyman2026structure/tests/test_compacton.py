import pytest, numpy as np
from kmm_compactons.compacton import universal_formulas, vk_criterion

class TestUniversalFormulas:
    @pytest.mark.parametrize("m,ea,el,ek", [(2,2.0,4.0,3/4),(3,1.0,3.0,2/3),(5,0.5,2.5,3/5)])
    def test_values(self, m, ea, el, ek):
        p = universal_formulas(m, A=1.0)
        assert abs(p.alpha - ea) < 1e-12
        assert abs(p.ell - el) < 1e-12
        assert abs(p.speed - ek) < 1e-12

    def test_ell_minus_alpha(self):
        for m in [2, 3, 4, 5, 7, 100]:
            p = universal_formulas(m)
            assert abs(p.ell - p.alpha - 2.0) < 1e-12

    def test_width_independence(self):
        for m in [2, 3, 5]:
            assert abs(universal_formulas(m, 0.1).W - universal_formulas(m, 10.0).W) < 1e-12

    def test_vk_positive(self):
        for m in [2, 3, 5]:
            assert vk_criterion(m, 1.0) > 0

    def test_profile_compact_support(self):
        p = universal_formulas(3, A=2.0)
        xi = np.linspace(-20, 20, 4096)
        assert np.all(p.evaluate(xi)[np.abs(xi) > p.W/2 + 1e-10] == 0.0)
