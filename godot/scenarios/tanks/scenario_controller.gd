extends Node

@export var tank:Tank
@export var target:Area3D
@export_category("RL Info")
@export var reward_target_reached = 1.0
@export var first_seen_reward = 0.1
@export var max_steps:= 500


var step_count := 0
var target_reached = false
var target_first_seen = false
var _original_tank_position:Transform3D

func _ready() -> void:
	target.body_entered.connect(on_target_body_entered)
	_original_tank_position = tank.transform
	
func on_target_body_entered(body):
	if body == tank:
		target_reached = true

func compute_reward_scenario():
	var reward = 0.0
	
	var observations = tank.get_observations()
	var target_visible = observations.get("target_visible", 0.0)
	if target_visible > 0.5 and not target_first_seen:
		target_first_seen = true
		reward += first_seen_reward
	
	if target_reached:
		reward += reward_target_reached
	
	return reward

func step(action:int):
	step_count+=1
	tank.apply_action(action)
	await get_tree().physics_frame

	var localReward = tank.get_reward()
	var scenarioReward = compute_reward_scenario()
	
	var truncated = step_count >= max_steps
	var done = target_reached or truncated
	
	return {
		"obs" : tank.get_observation_vector(),
		"reward" : localReward + scenarioReward,
		"done" : done,
		"info":{
			"local_term_rewards" : tank.get_reward_terms(),
			"scenario_reward" : scenarioReward,
			"target_reached" : target_reached,
			"target_first_seen" : target_first_seen
		}
	}
	
func reset_episode():
	target_reached = false
	target_first_seen = false
	step_count = 0
	tank.reset_reward()
	tank.transform = _original_tank_position
	return {
			"obs" : tank.get_observation_vector(),
			"info":{}
		}








	
