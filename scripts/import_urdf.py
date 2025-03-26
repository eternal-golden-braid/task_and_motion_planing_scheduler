from omni.isaac.core.utils import stage
from omni.isaac.core.robots import Robot
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.nucleus import get_assets_root_path

def import_ur10_to_scene():
    # Get the current stage
    current_stage = stage.get_current_stage()
    
    # Path to the UR10 URDF in Isaac Sim's assets
    assets_root_path = get_assets_root_path()
    ur10_path = assets_root_path + "/Isaac/Robots/Universal_Robots/ur10/ur10.usd"
    
    # Create a new prim for the robot
    robot_prim_path = "/World/ur10"
    
    # Add the UR10 reference to the stage
    add_reference_to_stage(usd_path=ur10_path, prim_path=robot_prim_path)
    
    # Create robot instance
    ur10_robot = Robot(
        prim_path=robot_prim_path,
        name="ur10",
        position=np.array([0, 0, 0]),  # Adjust position as needed
        orientation=np.array([1, 0, 0, 0])  # Adjust orientation as needed
    )
    
    return ur10_robot

# Usage example:
if __name__ == "__main__":
    # Make sure you're in a running Isaac Sim instance
    ur10 = import_ur10_to_scene()
    
    # Now you can manipulate the robot
    # For example, to set joint positions:
    # ur10.set_joint_positions(positions=[0, 0, 0, 0, 0, 0])