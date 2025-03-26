import omni
from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core_nodes.scripts.utils import create_urdf_from_stage

def export_to_urdf():
    # Get current stage
    stage = get_current_stage()
    
    # Specify your target prim path
    # This should match EXACTLY what you see in the Stage window
    target_prim_path = "/World/ur10e_robotiq2f_140"  # Replace YOUR_PRIM_NAME with your prim's name
    
    # Set export path
    output_path = "/home/leo/IsaacSimROS/multirobot_ws_2.0/urdf"  # Change this to your desired path
    
    try:
        # Use the core converter with target prim
        create_urdf_from_stage(
            output_directory=output_path,
            filename="robot.urdf",
            stage=stage,
            target_path=target_prim_path  # Add this line to specify target prim
        )
        print(f"URDF export completed for prim: {target_prim_path}")
    except Exception as e:
        print(f"Error during export: {e}")

if __name__ == "__main__":
    export_to_urdf()
