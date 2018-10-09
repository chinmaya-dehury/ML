'''
Description of this python code can be found here:

'''

import numpy as np

# Actions are: UP movement(0), DOWN movement(1), LEFT movement(2), RIGHT movement(3)

# R matrix
R = np.full((25,4),-1)



# reward to reach knight at state 6 is -100
R[1][1] = -100.0
R[5][3] = -100.0
R[7][2] = -100.0
R[11][0] = -100.0
# reward to reach knight at state 8 is -100
R[3][1] = -100.0
R[7][3] = -100.0
R[9][2] = -100.0
R[13][0] = -100.0
# reward to reach knight at state 12 is -100
R[7][1] = -100.0
R[11][3] = -100.0
R[13][2] = -100.0
R[17][0] = -100.0
# reward to reach knight at state 15 is -100
R[10][1] = -100.0
R[16][2] = -100.0
R[20][0] = -100.0
# reward to reach knight at state 19 is -100
R[14][1] = -100.0
R[18][3] = -100.0
R[24][0] = -100.0

# reward to reach the castle (prince) at state 22 is 100
R[17][1] = 100.0
R[21][3] = 100.0
R[23][2] = 100.0

# define the illegal movement
# -2 represent the illegal movement
illegal = -2
# illegal movement from states in FIRST ROW
R[0][0]=illegal
R[1][0]= illegal
R[2][0]= illegal
R[3][0]= illegal
R[4][0]= illegal
# illegal movement from states in FIRST COLUMN
R[0][2]= illegal
R[5][2]= illegal
R[10][2]= illegal
R[15][2]= illegal
R[20][2]= illegal
# illegal movement from states in LAST ROW
R[20][1]= illegal
R[21][1]= illegal
R[22][1]= illegal
R[23][1]= illegal
R[24][1]= illegal
# illegal movement from states in LAST COLUMN
R[4][3]= illegal
R[9][3]= illegal
R[14][3]= illegal
R[19][3]= illegal
R[24][3]= illegal

noOfState = 25
noOfAction = 4

# Q matrix
Q = np.zeros((noOfState,noOfAction))

gamma = 0.9
alpha = 0.1

# this function must exclude the illegal movement
def available_directions_to_go(state):
    curr_state_row = R[state,]
    act = np.where(curr_state_row != -2)
    return act[0]

def next_action_dir(all_psbl_directions):
    next_action_dir = int(np.random.choice(all_psbl_directions))
    return next_action_dir

def next_state(current_state, direction):
    if direction == 0: # UP direction
        st= current_state-5
    if direction == 1: # DOWN direction
        st = current_state+5
    if direction == 2: # LEFT direction
        st = current_state-1
    if direction == 3: # RIGHT direction
        st = current_state+1
    if verify_state(st) == False:
        print("6. Something wrong: (current_state, direction) ", current_state, direction)
    return st

def verify_state(st):
    if st < noOfState and st >= 0:
        return True
    else:
        return False

def verify_action(act):
    if act < noOfAction and act >= 0:
        return True
    else:
        return False

def updateQ(cur_state,action_dir,gamma):
    if verify_state(cur_state) == False or verify_action(action_dir)==False:
        print("3. something is Wrong")
    next_st = next_state(cur_state, action_dir)
    if verify_state(next_st) == False:
        print("4. something is Wrong: (next_st, cur_state, action_dir)",next_st, cur_state, action_dir)
        print(Q)
    index_of_max_val = np.where(R[next_st,] == np.max(R[next_st,]))[0][0]
    if verify_action(index_of_max_val)==False:
        print("5. something is Wrong")
    #print("next_st: ", next_st)
    #print("index_of_max_val:", index_of_max_val)
    #print("cur_state,action_dir,gamma", cur_state,action_dir,gamma)
    # Q[cur_state,action_dir] = R[cur_state, action_dir]+ gamma* Q[next_st,index_of_max_val]
    Q[cur_state, action_dir] = Q[cur_state, action_dir] + alpha*(R[cur_state, action_dir]
                                                                 + gamma*(Q[next_st,index_of_max_val] - Q[cur_state, action_dir]))


# --------------- T R A I N I N G ------------
for i in range(10000):
    cur_state = np.random.randint(0,noOfState)
    if verify_state(cur_state) == False:
        print("1. something is Wrong")
    next_act_dir = next_action_dir(available_directions_to_go(cur_state))
    if verify_action(next_act_dir) == False:
        print("2. something is Wrong")
    updateQ(cur_state, next_act_dir, gamma)
    if i%10000 == 0:
        print("10k iterations finished.")

# -------------- T E S T  I N G --------------

print("After training below is the Q matrix: ")
for i in range(Q.shape[0]):
        print(i,"  ",Q[i,])
start_st = 0
goal_st = 22
path = [start_st]
while start_st != goal_st:
    tmp = Q[start_st,]
    row_max = np.max(tmp[tmp != 0])
    max_reward_act_dir = np.where(Q[start_st,]==row_max)[0][0]
    next_st =  next_state(start_st, max_reward_act_dir)
    path.append(next_st)
    start_st = next_st

print("Path from state ",start_st," to state ",goal_st,": ",path)


###
#  Know more about knight and princess problem : https://medium.freecodecamp.org/diving-deeper-into-reinforcement-learning-with-q-learning-c18d0db58efe
###
