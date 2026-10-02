"""Regression for reader-scaled monopile loads and the rigid mudline grid."""
import unittest
import numpy as np
import pandas as pd

from weis.aeroelasticse.openmdao_qblade import QBLADELoadCases
from weis.aeroelasticse.QBlade_wrapper import QBladeWrapper


class TestMonopileLoading(unittest.TestCase):
    def test_tower_section_centres_keep_physical_sensor_coordinates(self):
        component = QBLADELoadCases(opt_options={
            'constraints': {'control': {'Max_TwrBsMyt': {'max': 1000.}}}})
        event = {}
        for axis in ['X', 'Y', 'Z']:
            for kind in ['For.', 'Mom.']:
                event[f'{axis}_tb {kind} TWR Bot. Constr.'] = 100.
                event[f'{axis}_tt {kind} TWR Top Constr.'] = 200.
                for index in range(1, 10):
                    event[f'{axis}_l {kind} TWR pos {index/10:.3f}'] = 100.+10.*index
        governing = 'Y_tb Mom. TWR Bot. Constr.'
        stats = pd.DataFrame({(f'{axis}_tb Mom. TWR Bot. Constr.', 'max'): [100.]
                              for axis in ['X', 'Y', 'Z']})
        outputs = {}
        component.get_tower_loading(stats, {governing: [event]},
                                    {'tower_z_full': np.array([15., 40., 115.])}, outputs)
        # Midpoints +27.5 and +77.5 m are at fractions .125 and .625.
        for axis in ['x', 'y', 'z']:
            np.testing.assert_allclose(outputs[f'tower_maxMy_F{axis}'], [112.5, 162.5])
            np.testing.assert_allclose(outputs[f'tower_maxMy_M{axis}'], [112.5, 162.5])

    def test_scaled_loads_map_to_exposed_section_heights(self):
        component = QBLADELoadCases()
        members = [1, 2, 3]
        positions = [0., 0., 1.]
        heights = [0., .5, 1.]
        component.qb_vt = {'QBladeOcean': {'SUB_Sensors': members,
                                         'SUB_Sensors_RelPos': positions}}
        component.Z_out_QBO_mpl = heights
        governing = 'Y_l Mom. SUB_member_0 pos 0.000'
        stats = pd.DataFrame({(governing, 'max'): [100.]})
        event = {}
        # Exercise the real reader conversion before interpolating the loads.
        reader = QBladeWrapper()
        for member, position, height in zip(members, positions, heights):
            for axis in ['X', 'Y', 'Z']:
                event[f'{axis}_l For. SUB_member_{member-1} pos {position:.3f}'] = reader.scale_channels(
                    np.array([1e6+1e5*height]), 'N')[0][0]
                event[f'{axis}_l Mom. SUB_member_{member-1} pos {position:.3f}'] = reader.scale_channels(
                    np.array([1e5+1e5*height]), 'Nm')[0][0]
        inputs = {'water_depth': np.array([40.]),
                  'monopile_z_full': np.array([-70., -40., -12.5, 15.])}
        outputs = {}
        component.get_monopile_loading(stats, {governing: [event]}, inputs, outputs)
        # Section centres: -55, -26.25, +1.25 m, hence exposed fractions 0,.25,.75.
        for axis in ['x', 'y', 'z']:
            np.testing.assert_allclose(outputs[f'monopile_maxMy_F{axis}'], [1000., 1025., 1075.])
            np.testing.assert_allclose(outputs[f'monopile_maxMy_M{axis}'], [100., 125., 175.])


if __name__ == '__main__':
    unittest.main()
