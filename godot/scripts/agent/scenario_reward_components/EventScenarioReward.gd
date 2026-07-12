extends "res://scripts/agent/scenario_reward_components/ScenarioRewardComponent.gd"
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
	if not bool(context.get(event_name, false)):
		return 0.0
	if only_once and bool(_rewarded_agents.get(agent_id, false)):
		return 0.0
	_rewarded_agents[agent_id] = true
	return reward * weight
