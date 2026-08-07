extends Node
class_name FunctionRewardComponent

enum EvaluationMode {
	CONTINUOUS,
	EPISODE_END
}

@export var enabled := true
@export var term_name := ""
@export var weight := 1.0
@export var evaluation_mode: EvaluationMode = EvaluationMode.CONTINUOUS
@export var method_name: StringName
@export var node_caller: Node
@export var required_event := ""
@export var pass_agent := false
@export var pass_context := false

var _episode_end_evaluated := {}


func get_term_name() -> String:
	return term_name if not term_name.is_empty() else str(name)


func reset_rewards() -> void:
	_episode_end_evaluated.clear()


func reset_agent(agent_id:String, _context:Dictionary = {}) -> void:
	_episode_end_evaluated.erase(agent_id)


func reset_reward(_context:Dictionary = {}) -> void:
	_episode_end_evaluated.clear()


func is_episode_end_only() -> bool:
	return evaluation_mode == EvaluationMode.EPISODE_END


func compute_reward(subject:Variant = null, context:Dictionary = {}) -> float:
	var reward_context := context
	if reward_context.is_empty() and typeof(subject) == TYPE_DICTIONARY:
		reward_context = subject

	if is_episode_end_only():
		if not _is_episode_end(reward_context):
			return 0.0
		var evaluation_key := _evaluation_key(subject, reward_context)
		if bool(_episode_end_evaluated.get(evaluation_key, false)):
			return 0.0
		_episode_end_evaluated[evaluation_key] = true
	if not required_event.is_empty() and not bool(_event_value(reward_context, required_event)):
		return 0.0
	if node_caller == null or method_name.is_empty() or not node_caller.has_method(method_name):
		return 0.0

	var arguments := []
	if pass_agent and subject is Node:
		arguments.append(subject)
	if pass_context:
		arguments.append(reward_context)
	var value: Variant = node_caller.callv(method_name, arguments)
	if typeof(value) == TYPE_INT or typeof(value) == TYPE_FLOAT:
		return float(value) * weight
	return 0.0


func _is_episode_end(context:Dictionary) -> bool:
	return (
		bool(context.get("episode_done", false))
		or bool(context.get("done", false))
		or bool(context.get("terminated", false))
		or bool(context.get("truncated", false))
		or bool(context.get("agent_terminal", false))
		or bool(context.get("event_terminal", false))
		or bool(context.get("scenario_terminal", false))
	)


func _evaluation_key(subject:Variant, context:Dictionary) -> String:
	var agent_id := str(context.get("agent_id", ""))
	if not agent_id.is_empty():
		return agent_id
	var agent: Variant = context.get("agent", null)
	if agent is Node:
		return str(agent.get_instance_id())
	var body: Variant = context.get("body", null)
	if body is Node:
		return str(body.get_instance_id())
	if subject is Node:
		return str(subject.get_instance_id())
	return "default"


func _event_value(context:Dictionary, event_name:String) -> Variant:
	if context.has(event_name):
		return context[event_name]
	var events: Variant = context.get("events", {})
	if typeof(events) == TYPE_DICTIONARY:
		return events.get(event_name, 0.0)
	return 0.0
