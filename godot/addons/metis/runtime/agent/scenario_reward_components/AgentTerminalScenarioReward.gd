extends "res://addons/metis/runtime/agent/scenario_reward_components/ScenarioRewardComponent.gd"
class_name AgentTerminalScenarioReward

@export var penalty := -60.0
@export var only_once := true

var _penalized_agents := {}


func reset_rewards() -> void:
	_penalized_agents.clear()


func reset_agent(agent_id:String, _context:Dictionary = {}) -> void:
	_penalized_agents[agent_id] = false


func compute_reward(_agent:Node, context:Dictionary = {}) -> float:
	var agent_id := str(context.get("agent_id", ""))
	if not bool(context.get("agent_terminal", false)):
		return 0.0
	if only_once and bool(_penalized_agents.get(agent_id, false)):
		return 0.0

	_penalized_agents[agent_id] = true
	return penalty * weight
