# A URDFLink that has a visual but no collision children
class_name URDFVisualNode
extends Node3D

@export var link: URDFLink

func update_link(
		link: URDFLink,
		robot_node: Node3D,
		owner_node: Node3D,
		options: Dictionary,
		source_path: String) -> void:
	self.link = link
	self.name = link.name
	
	robot_node.add_child(self)
	self.owner = owner_node

	for visual in link.visuals:
		var material = StandardMaterial3D.new()
		var c = visual.material_color
		if c != Vector4.ZERO:
			material.albedo_color = Color(c.x, c.y, c.z, c.w)
		elif visual.material_name in robot_node.urdf.materials:
			c = robot_node.urdf.materials[visual.material_name]
			material.albedo_color = Color(c.x, c.y, c.z, c.w)

		var gen = URDFGeometryFactory.get_visual_callable(int(visual.type))
		if gen:
			gen.call(
				self, owner_node, visual, options, source_path,
				null if c == Vector4.ZERO else material)
		else:
			push_error("Unknown type ", visual.type)
