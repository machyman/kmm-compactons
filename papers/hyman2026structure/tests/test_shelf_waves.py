import pytest, numpy as np
from kmm_compactons.shelf_waves import critical_speed, decay_rate, compacton_limit_k22

class TestShelfWaves:
    def test_critical_speed(self):
        assert abs(critical_speed(2, 1.0) - 2.0) < 1e-12
        assert abs(critical_speed(3, 1.0) - 3.0) < 1e-12
    def test_decay_rate(self):
        assert abs(decay_rate(2, 1.0, 6.0) - np.sqrt(2)) < 1e-12
    def test_compacton_limit(self):
        xi, f = compacton_limit_k22(4.0)
        assert abs(np.max(f) - 16/3) < 1e-6
