extends "res://scripts/agent/events/ScenarioEventSource.gd"
class_name AreaReachedEventSource

@export var area:Area3D
@export var only_once := true

var _agents := {}
var _reached := {}


func _ready() -> void:
	if area != null and not area.body_entered.is_connected(_on_body_entered):
		area.body_entered.connect(_on_body_entered)


func reset_events() -> void:
	_agents.clear()
	_reached.clear()


func reset_agent(agent:Node, context:Dictionary = {}) -> void:
	var agent_id := str(context.get("agent_id", agent.name))
	_agents[agent_id] = agent
	_reached[agent_id] = false


func get_agent_events(agent_id:String) -> Dictionary:
	return {event_name: bool(_reached.get(agent_id, false))}


func _on_body_entered(body:Node) -> void:
	for agent_id in _agents.keys():
		if _node_belongs_to_agent(body, _agents[agent_id]):
			if not only_once or not bool(_reached.get(agent_id, false)):
				_reached[agent_id] = true
			return


func _node_belongs_to_agent(node:Node, agent:Node) -> bool:
	var current := node
	while current != null:
		if current == agent:
			return true
		current = current.get_parent()
	return false
