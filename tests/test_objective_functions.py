import unittest
from ObjectiveFunction import get_property_value, ObjectiveFunctionClass  # Replace with actual class name
import os

class TestObjectiveFunction(unittest.TestCase):

    def setUp(self):
        # Setup code to create necessary objects and variables
        self.polymer = "example_polymer"
        self.xtb_opt_output_dir = "/path/to/xtb_opt_output_dir"
        self.host_ip = "127.0.0.1"
        self.InchiKey_initial = "example_inchikey_initial"
        self.data = [
            "Some line with cpu-time 1.0 h 30.0 min 45.0 s",
            "Some line with TOTAL ENERGY -123.456",
            "Some line with HOMO-LUMO GAP 1.234"
        ]


    def test_build_polymer(self):
        #run polymer building and make sure it works


    def test_get_property_value_total_energy(self):
        property_value = get_property_value(self.data, "TOTAL ENERGY")
        self.assertEqual(property_value, -123.456)

    def test_get_property_value_homo_lumo_gap(self):
        property_value = get_property_value(self.data, "HOMO-LUMO GAP")
        self.assertEqual(property_value, 1.234)

    def test_polymer_xtb_opt_calc(self):
        # Assuming ObjectiveFunctionClass is the class that contains the logic
        obj_func = ObjectiveFunctionClass(self.host_ip, self.xtb_opt_output_dir)
        polymer_xtb_opt_calc = obj_func.calculate_polymer_properties(self.polymer, self.InchiKey_initial)

        self.assertEqual(polymer_xtb_opt_calc["InChIKey"], "expected_inchikey")
        self.assertEqual(polymer_xtb_opt_calc["cal_folder"], os.path.join(self.xtb_opt_output_dir, "expected_inchikey"))
        self.assertEqual(polymer_xtb_opt_calc["Host IP"], self.host_ip)
        self.assertEqual(polymer_xtb_opt_calc["InChIKey_initial"], self.InchiKey_initial)
        self.assertEqual(polymer_xtb_opt_calc["cpu time"], "1.0 h 30.0 min 45.0 s")
        self.assertEqual(polymer_xtb_opt_calc["total energy (au)"], -123.456)
        self.assertEqual(polymer_xtb_opt_calc["HOMO-LUMO GAP (eV)"], 1.234)

if __name__ == '__main__':
    unittest.main()