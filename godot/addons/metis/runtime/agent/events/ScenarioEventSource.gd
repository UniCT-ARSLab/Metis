extends Metis
class_name ScenarioEventSource

@export var enabled := true
@export var event_name := "event"
@export var terminal_reason := ""


func reset_events() -> void:
	pass


func reset_agent(_agent:Node, _context:Dictionary = {}) -> void:
	pass


func update_agent(_agent:Node, _context:Dictionary = {}) -> void:
	pass


func get_agent_events(_agent_id:String) -> Dictionary:
	return {}


func get_agent_terminal_reason(agent_id:String) -> String:
	if terminal_reason.is_empty():
		return ""
	return terminal_reason if bool(get_agent_events(agent_id).get(event_name, false)) else ""
