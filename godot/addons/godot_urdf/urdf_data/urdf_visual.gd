class_name URDFVisual
extends Resource
# transform
@export var origin_xyz: Vector3
@export var origin_rpy: Vector3

@export var material_name: String
@export var material_color: Vector4 = Vector4.ZERO
@export var material_texture_path: String

@export var type: Type
# BOX
@export var size: Vector3
# SPHERE, CYLINDER
@export var radius: float
@export var length: float

# MESH
@export var mesh_path: String
@export var mesh_scale: Vector3

enum Type {BOX, MESH, CYLINDER, SPHERE}
