extends ScenarioEventSource
class_name ManualScenarioEventSource

var _current_values := {}
var _pending_values := {}


func trigger(agent_id:String, value:float = 1.0) -> void:
	_pending_values[agent_id] = float(_pending_values.get(agent_id, 0.0)) + value


func reset_events() -> void:
	_current_values.clear()
	_pending_values.clear()


func reset_agent(_agent:Node, context:Dictionary = {}) -> void:
	var agent_id := str(context.get("agent_id", ""))
	_current_values[agent_id] = 0.0
	_pending_values[agent_id] = 0.0


func update_agent(_agent:Node, context:Dictionary = {}) -> void:
	var agent_id := str(context.get("agent_id", ""))
	_current_values[agent_id] = float(_pending_values.get(agent_id, 0.0))
	_pending_values[agent_id] = 0.0


func get_agent_events(agent_id:String) -> Dictionary:
	return {event_name: float(_current_values.get(agent_id, 0.0))}
