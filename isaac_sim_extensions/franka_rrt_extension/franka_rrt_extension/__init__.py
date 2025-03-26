import omni.ext
import omni.ui as ui
import omni.kit.commands
import omni.graph.core as og
import carb
import numpy as np
import os

from pxr import Usd, UsdGeom, Gf

# Import our adapter
from .franka_rrt_adapter import FrankaRrtControllerAdapter

class FrankaRrtExtension(omni.ext.IExt):
    """Extension to connect RRT planner with ArticulationController in OmniGraph"""
    
    def on_startup(self, ext_id):
        print("[Franka RRT] Extension startup")
        self._window = ui.Window("Franka RRT Controller", width=300, height=300)
        
        self._rrt_adapter = None
        self._controller_node = None
        self._update_subscription = None
        
        # Build UI
        with self._window.frame:
            with ui.VStack(spacing=5, height=0):
                ui.Label("Franka RRT Controller")
                
                with ui.HStack(height=0):
                    ui.Label("Robot Path:", width=80)
                    self._robot_path_field = ui.StringField(height=0)
                    self._robot_path_field.model.set_value("/Root/franka_instanceable")
                
                with ui.HStack(height=0):
                    ui.Label("Target Path:", width=80)
                    self._target_path_field = ui.StringField(height=0)
                    self._target_path_field.model.set_value("/World/target")
                
                with ui.HStack(height=0):
                    ui.Label("Controller:", width=80)
                    self._controller_path_field = ui.StringField(height=0)
                    self._controller_path_field.model.set_value("/Graphs/Position_Controller/ArticulationController")
                
                ui.Spacer(height=10)
                with ui.HStack():
                    ui.Button("Initialize", clicked_fn=self._on_initialize)
                    ui.Button("Reset", clicked_fn=self._on_reset)
                    ui.Button("Disconnect", clicked_fn=self._on_disconnect)
    
    def on_shutdown(self):
        print("[Franka RRT] Extension shutdown")
        self._disconnect_adapter()
        if self._window:
            self._window = None
    
    def _on_initialize(self):
        """Initialize the adapter and connect to the controller"""
        robot_path = self._robot_path_field.model.get_value_as_string()
        target_path = self._target_path_field.model.get_value_as_string()
        controller_path = self._controller_path_field.model.get_value_as_string()
        
        # Create the adapter
        self._rrt_adapter = FrankaRrtControllerAdapter(
            robot_prim_path=robot_path,
            target_prim_path=target_path
        )
        
        # Initialize the adapter
        if not self._rrt_adapter.initialize():
            carb.log_error("Failed to initialize FrankaRrtControllerAdapter")
            return
        
        # Connect to the controller node
        self._connect_to_controller(controller_path)
        
        # Set up update callback
        self._setup_update_callback()
    
    def _on_reset(self):
        """Reset the adapter state"""
        if self._rrt_adapter:
            self._rrt_adapter.reset()
    
    def _on_disconnect(self):
        """Disconnect the adapter from the controller"""
        self._disconnect_adapter()
    
    def _connect_to_controller(self, controller_path):
        """Connect the adapter to the ArticulationController node"""
        if not self._rrt_adapter:
            carb.log_error("RRT adapter not initialized")
            return False
        
        # Get the controller node from the graph
        graph = og.Controller.get_default_graph_controller()
        self._controller_node = graph.get_node(controller_path)
        
        if not self._controller_node:
            carb.log_error(f"ArticulationController node not found at {controller_path}")
            return False
        
        # Connect adapter outputs to controller inputs
        robot_path = self._robot_path_field.model.get_value_as_string()
        
        # These operations should be performed through OmniGraph API
        # Set the targetPrim
        try:
            inputs = self._controller_node.get_attributes()
            
            # Find attribute IDs for inputs we need to set
            target_prim_attr_id = None
            joint_indices_attr_id = None
            joint_names_attr_id = None
            position_cmd_attr_id = None
            
            for attr_id, info in inputs.items():
                if info.name == "targetPrim":
                    target_prim_attr_id = attr_id
                elif info.name == "jointIndices":
                    joint_indices_attr_id = attr_id
                elif info.name == "jointNames":
                    joint_names_attr_id = attr_id
                elif info.name == "positionCommand":
                    position_cmd_attr_id = attr_id
            
            # Set attribute values
            if target_prim_attr_id:
                self._controller_node.set_attribute_value(target_prim_attr_id, robot_path)
            
            if joint_indices_attr_id:
                joint_indices = self._rrt_adapter.get_joint_indices()
                self._controller_node.set_attribute_value(joint_indices_attr_id, joint_indices)
            
            if joint_names_attr_id:
                joint_names = self._rrt_adapter.get_joint_names()
                self._controller_node.set_attribute_value(joint_names_attr_id, joint_names)
            
            # Initial position command
            if position_cmd_attr_id:
                pos_cmd = self._rrt_adapter.get_position_command()
                self._controller_node.set_attribute_value(position_cmd_attr_id, pos_cmd)
            
            print(f"[Franka RRT] Successfully connected to {controller_path}")
            return True
                
        except Exception as e:
            carb.log_error(f"Error connecting to controller: {e}")
            return False
    
    def _setup_update_callback(self):
        """Set up a callback to update the controller on each frame"""
        import omni.kit.app
        
        # Subscribe to update event
        self._update_subscription = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(
            self._on_update_frame
        )
    
    def _on_update_frame(self, event):
        """Update callback for each frame"""
        if not self._rrt_adapter or not self._controller_node:
            return
        
        # Update the adapter
        dt = event.payload["dt"]
        self._rrt_adapter.update(dt)
        
        # Update the controller node with new commands
        try:
            inputs = self._controller_node.get_attributes()
            
            # Find position command attribute
            position_cmd_attr_id = None
            for attr_id, info in inputs.items():
                if info.name == "positionCommand":
                    position_cmd_attr_id = attr_id
                    break
            
            if position_cmd_attr_id:
                pos_cmd = self._rrt_adapter.get_position_command()
                self._controller_node.set_attribute_value(position_cmd_attr_id, pos_cmd)
                
        except Exception as e:
            carb.log_error(f"Error updating controller: {e}")
    
    def _disconnect_adapter(self):
        """Disconnect the adapter from the controller"""
        if self._update_subscription:
            self._update_subscription = None
        
        self._controller_node = None
        self._rrt_adapter = None
        print("[Franka RRT] Disconnected from controller")