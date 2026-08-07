extends Node
class_name ScenarioRewardComponent

@export var enabled := true
@export var term_name := ""
@export var weight := 1.0
## Properties that Python scenario configuration may override at runtime. Keeping this
## list explicit prevents generic keys such as max_steps or training_mode from silently
## changing an unrelated reward property with the same name.
@export var scenario_config_properties: PackedStringArray = []


func get_term_name() -> String:
	if term_name.is_empty():
		return name
	return term_name


func apply_scenario_config(config:Dictionary) -> void:
	for property_name in scenario_config_properties:
		if not config.has(property_name):
			continue
		if _has_property(property_name):
			set(property_name, config[property_name])


func _has_property(property_name:String) -> bool:
	for property in get_property_list():
		if str(property.get("name", "")) == property_name:
			return true
	return false


func reset_rewards() -> void:
	pass


func reset_agent(_agent_id:String, _context:Dictionary = {}) -> void:
	pass


func compute_reward(_agent:Node, _context:Dictionary = {}) -> float:
	return 0.0


func is_agent_stalled(_agent_id:String) -> bool:
	return false


func get_terminal_reason(_agent_id:String) -> String:
	return ""


func get_agent_terms(_agent_id:String) -> Dictionary:
	return {}
