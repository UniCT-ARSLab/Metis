extends Node
class_name ScenarioRewardComponent

@export var enabled := true
@export var term_name := ""
@export var weight := 1.0


func get_term_name() -> String:
	if term_name.is_empty():
		return name
	return term_name


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
