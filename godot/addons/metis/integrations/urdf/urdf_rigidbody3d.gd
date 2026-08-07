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
		# Godot only accepts a manual center_of_mass when the mode is CUSTOM; without this it
		# rejects the assignment with "center_of_mass_mode != CENTER_OF_MASS_MODE_CUSTOM" for every
		# link whose URDF inertial origin is offset (e.g. OpenArm).
		self.center_of_mass_mode = RigidBody3D.CENTER_OF_MASS_MODE_CUSTOM
		self.center_of_mass = link.inertial.origin_xyz
