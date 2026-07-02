extends Node
class_name Agent

@export var actions : Dictionary = {}
@export var observations : Dictionary = {}
@export var reward_system_path: NodePath = NodePath("RewardSystem")

var _action_names: Array[String] = []
var _observation_names: Array[String] = []
var _reward_system: Node

func _ready() -> void:
	_reward_system = get_node_or_null(reward_system_path)

# AGENT'S ACTIONS
func add_action(action:String, callable:Callable) -> int:
	if actions.has(action):
		return ERR_ALREADY_EXISTS
	actions.set(action, callable)
	_action_names.append(action)
	return OK
	
func add_actions(new_actions:Dictionary) -> void:
	for action in new_actions.keys():
		self.add_action(str(action), new_actions[action])

func act(action:Variant) -> int:
	var action_name := ""

	if typeof(action) == TYPE_INT:
		var action_index := int(action)
		if action_index < 0 or action_index >= _action_names.size():
			return ERR_INVALID_PARAMETER
		action_name = _action_names[action_index]
	elif typeof(action) == TYPE_STRING or typeof(action) == TYPE_STRING_NAME:
		action_name = str(action)
	else:
		return ERR_INVALID_PARAMETER

	if not actions.has(action_name):
		return ERR_DOES_NOT_EXIST

	var callable: Callable = actions[action_name]
	if not callable.is_valid():
		return ERR_INVALID_DATA

	callable.call()
	return OK

func get_action_names() -> Array:
	return _action_names.duplicate()

func get_action_count() -> int:
	return _action_names.size()

# ANGET'S OBSERVABLE
func add_observation(observable:String, value:Variant = null) -> int:
	if observations.has(observable):
		return ERR_ALREADY_EXISTS
	observations.set(observable, value)
	_observation_names.append(observable)
	return OK
	
func add_observations(new_observations:Dictionary) -> void:
	for observable in new_observations.keys():
		self.add_observation(str(observable), new_observations[observable])
	
func update_observation(observable:String, new_value:Variant) -> int:
	if !observations.has(observable):
		return ERR_DOES_NOT_EXIST
	observations[observable] = new_value
	return OK

func get_observation_names() -> Array:
	return _observation_names.duplicate()
	
func get_observations() -> Dictionary:
	var resolved := {}
	for observable in _observation_names:
		resolved[observable] = _read_observation(observable)
	return resolved

func get_observation_vector() -> Array:
	var result := []
	for observable in _observation_names:
		_append_observation_value(result, _read_observation(observable))
	return result

func get_observation_size() -> int:
	return get_observation_vector().size()

# AGENT'S REWARD SYSTEM
func reset_reward(context:Dictionary = {}) -> void:
	var reward_system := _get_reward_system()
	if reward_system != null and reward_system.has_method("reset_reward"):
		reward_system.reset_reward(_build_reward_context(context))

func add_reward_event(term:String, value:float) -> void:
	var reward_system := _get_reward_system()
	if reward_system != null and reward_system.has_method("add_event"):
		reward_system.add_event(term, value)

func get_reward(context:Dictionary = {}) -> float:
	var reward_system := _get_reward_system()
	if reward_system == null or not reward_system.has_method("get_reward"):
		return 0.0

	return float(reward_system.get_reward(_build_reward_context(context)))

func get_reward_terms() -> Dictionary:
	var reward_system := _get_reward_system()
	if reward_system != null and reward_system.has_method("get_last_terms"):
		return reward_system.get_last_terms()

	return {}

# AGENT'S PRIVATE FUNCTIONS
func _read_observation(observable:String) -> Variant:
	if not observations.has(observable):
		return 0.0

	var source: Variant = observations[observable]
	if typeof(source) == TYPE_CALLABLE:
		var callable: Callable = source
		if callable.is_valid():
			return callable.call()
		return 0.0

	return source

func _get_reward_system() -> Node:
	if _reward_system == null:
		_reward_system = get_node_or_null(reward_system_path)
	return _reward_system

func _build_reward_context(context:Dictionary) -> Dictionary:
	var result := context.duplicate(true)

	if not result.has("agent"):
		result["agent"] = self
	if not result.has("body"):
		result["body"] = get_parent()
	if not result.has("observations"):
		result["observations"] = get_observations()

	return result

func _append_observation_value(result:Array, value:Variant) -> void:
	if value == null:
		result.append(0.0)
	elif typeof(value) == TYPE_BOOL:
		result.append(1.0 if value else 0.0)
	elif typeof(value) == TYPE_INT or typeof(value) == TYPE_FLOAT:
		result.append(float(value))
	elif typeof(value) == TYPE_VECTOR2:
		result.append(value.x)
		result.append(value.y)
	elif typeof(value) == TYPE_VECTOR3:
		result.append(value.x)
		result.append(value.y)
		result.append(value.z)
	elif typeof(value) == TYPE_ARRAY or typeof(value) == TYPE_PACKED_FLOAT32_ARRAY or typeof(value) == TYPE_PACKED_FLOAT64_ARRAY or typeof(value) == TYPE_PACKED_INT32_ARRAY or typeof(value) == TYPE_PACKED_INT64_ARRAY:
		for item in value:
			_append_observation_value(result, item)
	else:
		push_warning("Observation value cannot be flattened: %s" % [str(value)])
		result.append(0.0)
