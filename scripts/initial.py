import omni
from omni.isaac.core import SimulationContext
from omni.isaac.core.utils.prims import create_prim
from omni.isaac.core.utils.stage import get_current_stage

# Initialize the simulation context
simulation_context = SimulationContext()
stage = get_current_stage()

def main():
    # Your code goes here
    print("Hello from Isaac Sim!")

# Run the main function
if __name__ == "__main__":
    main()