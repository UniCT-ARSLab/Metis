extends Node
class_name ActionSpace


func get_action_space() -> Dictionary:
	var result := {}
	for child in get_children():
		if child.has_method("get_action_spec"):
			var spec: Variant = child.get_action_spec()
			if typeof(spec) == TYPE_DICTIONARY:
				var action_name := str(spec.get("name", child.name))
				spec.erase("name")
				result[action_name] = spec
	return result


func get_action_type() -> String:
	var action_space := get_action_space()
	var continuous_count := 0
	var discrete_count := 0
	for spec in action_space.values():
		var action_type := str(spec.get("action_type", "discrete"))
		if action_type == "continuous":
			continuous_count += 1
		elif action_type == "discrete":
			discrete_count += 1

	if continuous_count > 0 and discrete_count == 0:
		return "continuous"
	if discrete_count == 1 and continuous_count == 0 and action_space.size() == 1:
		return "discrete"
	return "hybrid"


func get_action_names() -> Array:
	var result := []
	for action_name in get_action_space().keys():
		var spec: Dictionary = get_action_space()[action_name]
		var action_type := str(spec.get("action_type", "discrete"))
		var size := int(spec.get("size", 1))
		if action_type == "discrete" and spec.has("names"):
			result.append_array(spec["names"])
		elif size <= 1:
			result.append(str(action_name))
		else:
			for idx in range(size):
				result.append("%s_%d" % [str(action_name), idx])
	return result


func get_continuous_action_size() -> int:
	var total := 0
	for spec in get_action_space().values():
		if str(spec.get("action_type", "discrete")) == "continuous":
			total += int(spec.get("size", 1))
	return total


func get_continuous_bounds(key:String, default_value:float) -> Array:
	var result := []
	for spec in get_action_space().values():
		if str(spec.get("action_type", "discrete")) != "continuous":
			continue
		var size := int(spec.get("size", 1))
		var value: Variant = spec.get(key, default_value)
		if typeof(value) == TYPE_ARRAY or typeof(value) == TYPE_PACKED_FLOAT32_ARRAY or typeof(value) == TYPE_PACKED_FLOAT64_ARRAY:
			var values: Array = Array(value)
			for idx in range(size):
				result.append(float(values[idx]) if idx < values.size() else default_value)
		else:
			for idx in range(size):
				result.append(float(value))
	return result


func execute_discrete_action(action:Variant, default_target:Node, component_name:String = "") -> int:
	var candidates := []
	for child in get_children():
		if not child.has_method("get_action_spec") or not child.has_method("execute_action"):
			continue
		var spec: Variant = child.get_action_spec()
		if typeof(spec) != TYPE_DICTIONARY or str(spec.get("action_type", "")) != "discrete":
			continue
		if not component_name.is_empty() and str(spec.get("name", child.name)) != component_name:
			continue
		candidates.append(child)

	if candidates.size() != 1:
		return ERR_UNAVAILABLE
	return int(candidates[0].execute_action(action, default_target))
