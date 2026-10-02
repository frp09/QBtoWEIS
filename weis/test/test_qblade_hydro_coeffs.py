import unittest

from weis.aeroelasticse.openmdao_qblade import QBLADELoadCases


class TestQBladeHydroCoeffNormalization(unittest.TestCase):

    def _component(self):
        # The helper methods do not depend on OpenMDAO setup state.
        return QBLADELoadCases()

    def test_fixed_bottom_legacy_scalar_is_autoconverted(self):
        comp = self._component()
        qb_vt = {
            'QBladeOcean': {
                'override_morison_coefficients': True,
                'HydroCdN': 1.0,
                'HydroCaN': 1.0,
                'HydroCpN': 0.0,
                'HydroCdA': 1.6,
                'HydroCaA': 1.0,
                'HydroCpA': 0.0,
            }
        }
        modopt = {'flags': {'floating': False}}

        comp._normalize_qblade_ocean_hydro_coefficients(qb_vt, modopt)

        self.assertEqual(qb_vt['QBladeOcean']['HydroCdN']['monopile'], [1.0, 1.0])
        self.assertEqual(qb_vt['QBladeOcean']['HydroCpA']['monopile'], [0.0, 0.0])

    def test_fixed_bottom_single_dict_key_is_remapped_to_monopile(self):
        comp = self._component()
        qb_vt = {
            'QBladeOcean': {
                'override_morison_coefficients': True,
                'HydroCdN': {'pile': [1.2, 1.2]},
                'HydroCaN': {'pile': [0.9, 0.9]},
                'HydroCpN': {'pile': [0.0, 0.0]},
                'HydroCdA': {'pile': [1.6, 1.6]},
                'HydroCaA': {'pile': [1.0, 1.0]},
                'HydroCpA': {'pile': [0.0, 0.0]},
            }
        }
        modopt = {'flags': {'floating': False}}

        comp._normalize_qblade_ocean_hydro_coefficients(qb_vt, modopt)

        self.assertEqual(qb_vt['QBladeOcean']['HydroCdN']['monopile'], [1.2, 1.2])
        self.assertNotIn('pile', qb_vt['QBladeOcean']['HydroCdN'])


if __name__ == '__main__':
    unittest.main()
