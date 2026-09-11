@tool
class_name URDFRobot
extends MetisURDFResource

@export var name: String
@export var links: Array[URDFLink] = []
@export var joints: Array[URDFJoint] = []
@export var materials: Dictionary[String, Vector4] = {}

func get_child_joints(link_name: String) -> Array[URDFJoint]:
	var children: Array[URDFJoint] = []
	for joint in joints:
		if joint.parent == link_name:
			children.append(joint)
	return children

func get_joint(joint_name: String) -> URDFJoint:
	for joint in joints:
		if joint.name == joint_name:
			return joint
	return null

func get_mimic_joints() -> Array[URDFJoint]:
	var result: Array[URDFJoint] = []
	for joint in joints:
		if joint.is_mimic():
			result.append(joint)
	return result

func get_actuated_joints() -> Array[URDFJoint]:
	var result: Array[URDFJoint] = []
	for joint in joints:
		if (
			joint.type in ["revolute", "continuous", "prismatic"]
			and not joint.is_mimic()
		):
			result.append(joint)
	return result

func get_actuated_joint_names() -> PackedStringArray:
	var result := PackedStringArray()
	for joint in get_actuated_joints():
		result.append(joint.name)
	return result

func validate_mimic_joints() -> PackedStringArray:
	var errors := PackedStringArray()
	var joints_by_name: Dictionary[String, URDFJoint] = {}
	for joint in joints:
		joints_by_name[joint.name] = joint

	for joint in get_mimic_joints():
		if not joints_by_name.has(joint.mimic_joint):
			errors.append(
				"Joint '%s' mimics unknown joint '%s'." %
				[joint.name, joint.mimic_joint])
			continue

		var visited: Dictionary[String, bool] = {}
		var current: URDFJoint = joint
		while current != null and current.is_mimic():
			if visited.has(current.name):
				errors.append("Mimic cycle detected at joint '%s'." % current.name)
				break
			visited[current.name] = true
			current = joints_by_name.get(current.mimic_joint)
	return errors
	
func get_link(link_name: String) -> URDFLink:
	for link in links:
		if link.name == link_name:
			return link
	return null

func get_root_links() -> Array[String]:
	var roots: Array[String] = []
	for link in links:
		var has_parent = false
		for joint in joints:
			if joint.child == link.name:
				has_parent = true
				break
		if not has_parent:
			roots.append(link.name)
	return roots
