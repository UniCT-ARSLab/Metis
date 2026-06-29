extends Node
class_name Agent

@export var actions : Dictionary = {}
@export var observations : Dictionary = {}

func add_action(action:String, callable:Callable):
	if actions.has(action):
		return Error.ERR_ALREADY_EXISTS
	actions.set(action, callable)
	
func add_actions(actions:Dictionary[String,Callable]):
	for action in actions.keys():
		self.add_action(action, actions[action])

func add_observation(observable:String, value:Variant = null):
	if observations.has(observable):
		return Error.ERR_ALREADY_EXISTS
	observations.set(observable, value)
	
func add_observations(observations:Dictionary[String,Variant]):
	for observable in observations.keys():
		self.add_observation(observable, observations[observable])
	
func update_observation(observable:String, newValue:Variant):
	if !observations.has(observable):
		return Error.ERR_DOES_NOT_EXIST
	observations[observable] = newValue
	
func get_observations():
	return observations
