class_name URDFInertial
extends Resource

@export var mass: float = 1.0 # Default to 1kg if missing to prevent physics errors
@export var inertia_tensor: Dictionary = {} # (Optional)
@export var origin_xyz: Vector3
@export var origin_rpy: Vector3
