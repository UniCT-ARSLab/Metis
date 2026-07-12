extends "res://scripts/agent/events/ScenarioEventSource.gd"
class_name ObservationThresholdEventSource

@export var observation_name := ""
@export var threshold := 0.5
@export var trigger_when_greater := true
@export var latch := true

var _triggered := {}


func reset_events() -> void:
	_triggered.clear()


func reset_agent(agent:Node, context:Dictionary = {}) -> void:
	var agent_id := str(context.get("agent_id", agent.name))
	_triggered[agent_id] = false


func update_agent(agent:Node, context:Dictionary = {}) -> void:
	var agent_id := str(context.get("agent_id", agent.name))
	if latch and bool(_triggered.get(agent_id, false)):
		return
	if not agent.has_method("get_observations"):
		_triggered[agent_id] = false
		return
	var observations:Dictionary = agent.get_observations()
	var value := float(observations.get(observation_name, 0.0))
	_triggered[agent_id] = value >= threshold if trigger_when_greater else value <= threshold


func get_agent_events(agent_id:String) -> Dictionary:
	return {event_name: bool(_triggered.get(agent_id, false))}
