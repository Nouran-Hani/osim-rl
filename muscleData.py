# import pandas as pd

# # Load the raw .sto file
# input_path = "D:\\Projects\\osim-rl\\gait14dof22musc_StaticOptimization_activation.sto"
# output_path = "D:\\Projects\\osim-rl\\muscle_activations.csv"

# # Load skipping the OpenSim header
# data = pd.read_csv(input_path, sep='\t', skiprows=8)

# # Clean column names (e.g., convert '/forceset/vasti_r' to 'vasti_r')
# data.columns = [c.split('/')[-1] for c in data.columns]

# # Save as CSV
# data.to_csv(output_path, index=False)
# print(f"File converted and saved to: {output_path}")


import numpy as np
from osim.env import L2M2019Env

env = L2M2019Env(visualize=False, difficulty=0)
env.reset()

# Get the opensim model object
model = env.osim_model.model
state = env.osim_model.state

# Extract actuator names in the exact order the action space expects
actuators = model.getActuators()
muscle_names_in_order = []
for i in range(actuators.getSize()):
    muscle_names_in_order.append(actuators.get(i).getName())

print(f"Total actuators: {len(muscle_names_in_order)}")
for i, name in enumerate(muscle_names_in_order):
    print(f"  [{i}] {name}")

env.close()


import pandas as pd

csv_path = "D:\\Projects\\osim-rl\\muscle_activations.csv"
data = pd.read_csv(csv_path)

print("\n--- Validation ---")
missing_in_csv   = [m for m in muscle_names_in_order if m not in data.columns]
extra_in_csv     = [c for c in data.columns if c not in muscle_names_in_order]
order_match      = list(data.columns) == muscle_names_in_order

print(f"Order matches exactly: {order_match}")
print(f"Missing from CSV:      {missing_in_csv}")
print(f"Extra cols in CSV:     {extra_in_csv}")