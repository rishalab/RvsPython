from pyJoules.energy_meter import measure_energy
from pyJoules.handler.csv_handler import CSVHandler

csv_handler = CSVHandler('output_test6.csv')

@measure_energy(handler=csv_handler)
def test_function():
    return 

for i in range(10):
    test_function()

print("Process complete")
csv_handler.save_data()