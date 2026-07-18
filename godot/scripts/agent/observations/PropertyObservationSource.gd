@tool
extends "res://scripts/agent/observations/ObservationSource.gd"
class_name PropertyObservationSource

@export var observation_name := ""
@export var source_path: NodePath
@export var property_path: NodePath

@export_category("Numeric normalization")
@export var normalize_numeric := false
@export var input_min := 0.0
@export var input_max := 1.0
@export var output_min := 0.0
@export var output_max := 1.0
@export var clamp_normalized := true

var _source: Node
var _last_value: Variant = 0.0


func register_observations(agent:Agent, body:Node) -> void:
	if observation_name.is_empty() or property_path.is_empty():
		return

	_source = _resolve_source(body)
	if _source == null:
		push_warning("Observation property source not found: %s" % source_path)
		return
	if not _has_root_property(_source, property_path):
		push_warning("Observation property not found: %s" % property_path)
		return

	var initial_value: Variant = _source.get_indexed(property_path)
	if initial_value == null:
		push_warning("Observation property is null or unreadable: %s" % property_path)
		return
	_last_value = _transform_value(initial_value)
	agent.add_observation(observation_name, Callable(self, "_read_property"))


func get_editor_source() -> Node:
	return _resolve_source(_editor_body())


func _read_property() -> Variant:
	if _source == null:
		return _last_value
	var value: Variant = _source.get_indexed(property_path)
	if value == null:
		return _last_value
	_last_value = _transform_value(value)
	return _last_value


func _transform_value(value:Variant) -> Variant:
	if not normalize_numeric or (typeof(value) != TYPE_INT and typeof(value) != TYPE_FLOAT):
		return value

	var denominator := input_max - input_min
	if absf(denominator) <= 0.000001:
		return output_min
	var ratio := (float(value) - input_min) / denominator
	if clamp_normalized:
		ratio = clampf(ratio, 0.0, 1.0)
	return lerpf(output_min, output_max, ratio)


func _resolve_source(body:Node) -> Node:
	if body == null or source_path.is_empty():
		return body
	return body.get_node_or_null(source_path)


func _editor_body() -> Node:
	var observation_system := get_parent()
	var agent := observation_system.get_parent() if observation_system != null else null
	return agent.get_parent() if agent != null else null


func _has_root_property(source:Node, path:NodePath) -> bool:
	if path.get_name_count() <= 0:
		return false
	var root_name := str(path.get_name(0))
	for property in source.get_property_list():
		if str(property.get("name", "")) == root_name:
			return true
	return false
