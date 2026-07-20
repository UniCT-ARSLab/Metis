class_name URDFRigidBody3D
extends RigidBody3D

@export var link: URDFLink

func update_link(
		link: URDFLink,
		robot_node: Node3D,
		owner_node: Node3D,
		options: Dictionary,
		source_path: String) -> void:
	self.link = link
	URDFPhysicsBody3D.update_link(
		self, link, robot_node, owner_node, options, source_path)
	if link.inertial:
		self.center_of_mass = link.inertial.origin_xyz
