extends ScenarioRewardComponent
class_name EventScenarioReward

@export var event_name := "event"
@export var reward := 5.0
@export var only_once := true

var _rewarded_agents := {}


func reset_rewards() -> void:
	_rewarded_agents.clear()


func reset_agent(agent_id:String, _context:Dictionary = {}) -> void:
	_rewarded_agents[agent_id] = false


func compute_reward(_agent:Node, context:Dictionary = {}) -> float:
	var agent_id := str(context.get("agent_id", ""))
	var event_value: Variant = context.get(event_name, 0.0)
	if not bool(event_value):
		return 0.0
	if only_once and bool(_rewarded_agents.get(agent_id, false)):
		return 0.0
	_rewarded_agents[agent_id] = true
	var multiplier := float(event_value) if typeof(event_value) == TYPE_INT or typeof(event_value) == TYPE_FLOAT else 1.0
	return reward * multiplier * weight
