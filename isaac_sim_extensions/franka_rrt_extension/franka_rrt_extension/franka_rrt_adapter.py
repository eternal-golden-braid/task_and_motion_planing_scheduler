import omni.usd
import omni.timeline
import numpy as np
import carb

from pxr import Usd, UsdGeom, Gf
from isaacsim.core.utils.extensions import get_extension_path_from_name
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.prims import Articulation
from isaacsim.core.utils.nucleus import get_assets_root_path
from isaacsim.core.api.objects.cuboid import VisualCuboid
from isaacsim.core.prims import XFormPrim
from isaacsim.core.utils.numpy.rotations import euler_angles_to_quats, quats_to_rot_matrices
from isaacsim.core.utils.distance_metrics import rotational_distance_angle

from isaacsim.robot_motion.motion_generation import PathPlannerVisualizer
from isaacsim.robot_motion.motion_generation.lula import RRT
from isaacsim.robot_motion.motion_generation import interface_config_loader

class FrankaRrtControllerAdapter:
    """
    Adapter class to connect RRT Planner outputs to ArticulationController inputs.
    """
    def __init__(self, 
                 robot_prim_path="/Root/franka_instanceable",
                 target_prim_path="/World/target",
                 obstacle_prim_path="/World/Wall"):
        self._robot_prim_path = robot_prim_path
        self._target_prim_path = target_prim_path
        self._obstacle_prim_path = obstacle_prim_path
        
        self._rrt = None
        self._path_planner_visualizer = None
        self._plan = []
        
        self._articulation = None
        self._target = None
        self._target_position = None
        
        self._joint_names = []
        self._joint_indices = []
        self._position_command = []
        self._velocity_command = []
        self._effort_command = []
        
        self._frame_counter = 0
        self._target_translation = np.zeros(3)
        self._target_rotation = np.eye(3)
        
        self._initialized = False
        
    def initialize(self):
        """Initialize the RRT planner and connect to the robot"""
        # Get the articulation from the stage
        self._articulation = Articulation(self._robot_prim_path)
        
        # Handle the target and obstacle (assuming they exist or create them)
        stage = omni.usd.get_context().get_stage()
        target_prim = stage.GetPrimAtPath(self._target_prim_path)
        if not target_prim.IsValid():
            add_reference_to_stage(get_assets_root_path() + "/Isaac/Props/UIElements/frame_prim.usd", self._target_prim_path)
        self._target = XFormPrim(self._target_prim_path, scale=[.04,.04,.04])
        
        # Setup the RRT planner
        mg_extension_path = get_extension_path_from_name("isaacsim.robot_motion.motion_generation")
        rmp_config_dir = os.path.join(mg_extension_path, "motion_policy_configs")
        rrt_config_dir = os.path.join(mg_extension_path, "path_planner_configs")
        
        # Initialize an RRT object
        self._rrt = RRT(
            robot_description_path = rmp_config_dir + "/franka/rmpflow/robot_descriptor.yaml",
            urdf_path = rmp_config_dir + "/franka/lula_franka_gen.urdf",
            rrt_config_path = rrt_config_dir + "/franka/rrt/franka_planner_config.yaml",
            end_effector_frame_name = "right_gripper"
        )
        
        # Add obstacle if it exists
        obstacle_prim = stage.GetPrimAtPath(self._obstacle_prim_path)
        if obstacle_prim.IsValid():
            self._rrt.add_obstacle(obstacle_prim)
        
        # Set the maximum number of iterations
        self._rrt.set_max_iterations(5000)
        
        # Create the visualizer
        self._path_planner_visualizer = PathPlannerVisualizer(self._articulation, self._rrt)
        
        # Get joint names and indices from the articulation
        self._joint_names = self._articulation.dof_names
        self._joint_indices = list(range(len(self._joint_names)))
        
        # Initialize commands with zeros
        num_dofs = self._articulation.num_dof
        self._position_command = np.zeros(num_dofs)
        self._velocity_command = np.zeros(num_dofs)
        self._effort_command = np.zeros(num_dofs)
        
        self._initialized = True
        self.reset()
        return True
    
    def update(self, dt):
        """Update the controller and generate new commands"""
        if not self._initialized:
            carb.log_error("FrankaRrtControllerAdapter not initialized!")
            return False
            
        current_target_translation, current_target_orientation = self._target.get_world_pose()
        current_target_rotation = quats_to_rot_matrices(current_target_orientation)
        
        translation_distance = np.linalg.norm(self._target_translation - current_target_translation)
        rotation_distance = rotational_distance_angle(current_target_rotation, self._target_rotation)
        target_moved = translation_distance > 0.01 or rotation_distance > 0.01
        
        if (self._frame_counter % 60 == 0 and target_moved):
            # Replan every 60 frames if the target has moved
            self._rrt.set_end_effector_target(current_target_translation, current_target_orientation)
            self._rrt.update_world()
            self._plan = self._path_planner_visualizer.compute_plan_as_articulation_actions(max_cspace_dist=.01)
            
            self._target_translation = current_target_translation
            self._target_rotation = current_target_rotation
        
        if self._plan:
            # Get the next action from the plan
            action = self._plan.pop(0)
            
            # Extract position commands from action
            self._position_command = action.joint_positions
            self._velocity_command = action.joint_velocities if hasattr(action, 'joint_velocities') else np.zeros_like(self._position_command)
            self._effort_command = action.joint_efforts if hasattr(action, 'joint_efforts') else np.zeros_like(self._position_command)
        
        self._frame_counter += 1
        return True
    
    def reset(self):
        """Reset the controller state"""
        self._target_translation = np.zeros(3)
        self._target_rotation = np.eye(3)
        self._frame_counter = 0
        self._plan = []
        
        # Reset commands to zeros or initial pose
        num_dofs = len(self._joint_indices)
        self._position_command = np.zeros(num_dofs)
        self._velocity_command = np.zeros(num_dofs)
        self._effort_command = np.zeros(num_dofs)
    
    # Getters for ArticulationController inputs
    def get_joint_names(self):
        return self._joint_names
    
    def get_joint_indices(self):
        return self._joint_indices
    
    def get_position_command(self):
        return self._position_command
    
    def get_velocity_command(self):
        return self._velocity_command
    
    def get_effort_command(self):
        return self._effort_command


# Example of connecting to the ArticulationController in OmniGraph
def connect_to_articulation_controller():
    """Connect the RRT adapter to an ArticulationController OmniGraph node"""
    import omni.graph.core as og
    
    # Create the adapter
    adapter = FrankaRrtControllerAdapter(
        robot_prim_path="/Root/franka_instanceable",
        target_prim_path="/World/target"
    )
    
    # Initialize the adapter
    if not adapter.initialize():
        carb.log_error("Failed to initialize FrankaRrtControllerAdapter")
        return None
    
    # Get the controller node from the graph
    # Note: You need to provide the correct graph and node paths
    graph = og.Controller.get_default_graph_controller()
    node_path = "/Graphs/Position_Controller/ArticulationController"
    controller_node = graph.get_node(node_path)
    
    if not controller_node:
        carb.log_error(f"ArticulationController node not found at {node_path}")
        return None
    
    # Connect the adapter outputs to the controller inputs
    # These operations should be performed through OmniGraph API
    # This is pseudocode - actual implementation depends on specific OmniGraph usage
    controller_node.set_attribute_value("targetPrim", "/Root/franka_instanceable")
    controller_node.set_attribute_value("jointIndices", adapter.get_joint_indices())
    controller_node.set_attribute_value("jointNames", adapter.get_joint_names())
    
    # You would need to set up a callback or subscription to update commands
    # from the adapter to the controller in each frame
    
    return adapter

# Example of using a callback to update controller inputs
def update_controller(adapter, controller_node):
    """Update the controller inputs from the adapter"""
    # Get the latest commands from the adapter
    position_command = adapter.get_position_command()
    
    # Update the controller node
    controller_node.set_attribute_value("positionCommand", position_command)
    
    # Optionally set velocity and effort commands if using those control modes
    # controller_node.set_attribute_value("velocityCommand", adapter.get_velocity_command())
    # controller_node.set_attribute_value("effortCommand", adapter.get_effort_command())