extends Metis
class_name ScenarioRewardSystem

var _last_component_values := {}


func reset_rewards() -> void:
	_last_component_values.clear()
	for component in _scenario_components():
		if component.has_method("reset_rewards"):
			component.reset_rewards()


func reset_agent(agent_id:String, context:Dictionary = {}) -> void:
	_last_component_values[agent_id] = {}
	var agent_context := context.duplicate(true)
	agent_context["agent_id"] = agent_id
	for component in _scenario_components():
		if component.has_method("reset_agent"):
			component.reset_agent(agent_id, agent_context)


func compute_reward_scenario(agent:Node, context:Dictionary = {}) -> float:
	var agent_id := str(context.get("agent_id", agent.name))
	var reward_context := context.duplicate(true)
	var total := 0.0
	var values := {}
	var episode_end_components := []
	for component in _scenario_components():
		if component.has_method("is_episode_end_only") and bool(component.is_episode_end_only()):
			episode_end_components.append(component)
			continue
		total += _compute_component(component, agent, reward_context, values)

	var scenario_terminal_reason := get_terminal_reason(agent_id)
	if not scenario_terminal_reason.is_empty():
		reward_context["scenario_terminal"] = true
		reward_context["scenario_terminal_reason"] = scenario_terminal_reason
	var terminated := (
		bool(reward_context.get("terminated", false))
		or bool(reward_context.get("agent_terminal", false))
		or bool(reward_context.get("event_terminal", false))
		or bool(reward_context.get("scenario_terminal", false))
	)
	var episode_done := terminated or bool(reward_context.get("truncated", false))
	reward_context["terminated"] = terminated
	reward_context["done"] = episode_done
	reward_context["episode_done"] = episode_done

	for component in episode_end_components:
		total += _compute_component(component, agent, reward_context, values)
	_last_component_values[agent_id] = values
	return total


func get_terminal_reason(agent_id:String) -> String:
	for component in _scenario_components():
		if not component.has_method("get_terminal_reason"):
			continue
		var reason := str(component.get_terminal_reason(agent_id))
		if not reason.is_empty():
			return reason
	return ""


func is_agent_stalled(agent_id:String) -> bool:
	for component in _scenario_components():
		if component.has_method("is_agent_stalled") and bool(component.is_agent_stalled(agent_id)):
			return true
	return false


func get_agent_terms(agent_id:String) -> Dictionary:
	var result := {
		"component_values": Dictionary(_last_component_values.get(agent_id, {})).duplicate(true)
	}
	for component in _scenario_components():
		if not component.has_method("get_agent_terms"):
			continue
		var terms:Variant = component.get_agent_terms(agent_id)
		if typeof(terms) != TYPE_DICTIONARY or terms.is_empty():
			continue
		var term_name := str(component.get_term_name() if component.has_method("get_term_name") else component.name)
		result[term_name] = Dictionary(terms).duplicate(true)
	return result


func _scenario_components() -> Array:
	var result := []
	for child in get_children():
		if child.get("enabled") == false:
			continue
		if child.has_method("compute_reward"):
			result.append(child)
	return result


func _compute_component(component:Node, agent:Node, context:Dictionary, values:Dictionary) -> float:
	var value := float(component.compute_reward(agent, context))
	var term_name := str(component.get_term_name() if component.has_method("get_term_name") else component.name)
	values[term_name] = value
	return value
