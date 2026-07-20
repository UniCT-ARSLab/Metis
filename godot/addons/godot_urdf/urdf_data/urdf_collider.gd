class_name URDFCollider
extends Resource

# All XYZ will be kept as is originally in URDF file
# Y and Z should be flipped when generating Nodes

@export var origin_xyz: Vector3
@export var origin_rpy: Vector3

@export var type: Type
@export var size: Vector3
@export var radius: float
@export var length: float

@export var mesh_path: String

enum Type {BOX, MESH, CYLINDER, SPHERE}
