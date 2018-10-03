import numpy as np

nos = 8 # number of states
goal_state = 7 # the goal state
intermediate_Q_mat_at_step=1000

# R matrix
R = np.matrix([ [-1,0,-1,-1,-1,-1,-1,100],
		[0,-1,0,-1,-1,-1,-1,-1],
		[-1,0,-1,0,-1,-1,-1,100],
		[-1,-1,0,-1,0,-1,-1,-1],
		[-1,-1,-1,0,-1,0,-1,-1],
		[-1,-1,-1,-1,0,-1,0,-1],
		[-1,-1,-1,-1,-1,0,-1,100],
		[0,-1,0,-1,-1,-1,0,100] ])

# Q matrix
Q = np.matrix(np.zeros([nos,nos]))
# Gamma (learning factor)
gamma = 0.8

def available_actions(state):
    current_state_row = R[state,]
    av_act = np.where(current_state_row >= 0)[1]
    return av_act


# A random action is selected
def sample_next_action(available_actions_range):
    next_action = int(np.random.choice(available_act,1))
    return next_action


# update the Q-matrix for training phase
def update(current_state, action, gamma):
    max_index = np.where(Q[action,] == np.max(Q[action,]))[1]
    if max_index.shape[0] > 1:
        max_index = int(np.random.choice(max_index, size = 1))
    else:
        max_index = int(max_index)        
    # Q learning formula
    Q[current_state, action] = R[current_state, action] + gamma *  Q[action, max_index]
    
############ Training stage ###################################

# Train for 10000 iterations.
for i in range(10000): 
    current_state = np.random.randint(0, int(Q.shape[0]))
    available_act = available_actions(current_state)
    action = sample_next_action(available_act)
    update(current_state,action,gamma)
    # print the intermediate Q matrix
    if((i % intermediate_Q_mat_at_step) == 0):
        print("Q-matrix at step ", i)
        print(Q/np.max(Q)*100)
    
# Normalize the "trained" Q matrix
print("Trained Q matrix:")
print(Q/np.max(Q)*100)

###################### Testing Stage  ###################################

current_state = 4
steps = [current_state]

while current_state != goal_state:

    next_step_index = np.where(Q[current_state,] == np.max(Q[current_state,]))[1]
    
    if next_step_index.shape[0] > 1:
        next_step_index = int(np.random.choice(next_step_index, size = 1))
    else:
        next_step_index = int(next_step_index)
    
    steps.append(next_step_index)
    current_state = next_step_index

# Print selected sequence of steps
print("Selected path:")
print(steps)
