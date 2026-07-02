extends RewardComponent
class_name StepPenaltyReward

@export var penalty := -0.001


func compute_reward(_context:Dictionary) -> float:
	return penalty * weight
