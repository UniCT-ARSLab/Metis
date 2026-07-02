extends Node
class_name ScenarioRewardSystem

var _agent_events := {}


func reset_rewards() -> void:
	_agent_events.clear()


func add_agent_event(agent_id:String, term:String, value:float) -> void:
	if not _agent_events.has(agent_id):
		_agent_events[agent_id] = {}

	var terms: Dictionary = _agent_events[agent_id]
	terms[term] = float(terms.get(term, 0.0)) + value


func compute_rewards(agent_ids:Array, context:Dictionary = {}) -> Dictionary:
	var rewards := {}

	for agent_id_value in agent_ids:
		var agent_id := str(agent_id_value)
		var terms := {}
		var total := 0.0

		if _agent_events.has(agent_id):
			for term in _agent_events[agent_id].keys():
				var value := float(_agent_events[agent_id][term])
				terms[term] = value
				total += value

		rewards[agent_id] = {
			"reward": total,
			"terms": terms
		}

	_agent_events.clear()
	return rewards
