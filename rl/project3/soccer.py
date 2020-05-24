"""
environment for soccer game described in "Correlated-Q Learning"
by Amy Greenwald and Keith Hall
"""
import numpy as np
from util import categorical_sample



class soccer(object):

    def __init__(self):
        self.nS = 128
        nR = 2
        nC = 4
        maxR = nR-1
        maxC = nC-1
        """
        Actions:    stick, N, S, E, W
        corresponds to: 0, 1, 2, 3, 4
        """
        nA = 5
        self.nJointA = nA ** 2
        self.nA = nA
        # set up transition P
        P = {s : {a : [] for a in range(nA**2)} for s in range(self.nS)}
        for col_A in range(4):
            for col_B in range(4):
                for row_A in range(2):
                    for row_B in range (2):
                        if col_A == col_B and row_A == row_B:
                            pass
                        else:
                            for ball in [-1,1]:
                                state = self.encode_state(col_A, col_B, row_A, row_B, ball)
                                for a_A in range(nA):
                                    for a_B in range(nA):
                                        new_col_A, new_col_B, new_row_A, new_row_B, new_ball = col_A, col_B, row_A, row_B, ball
                                        r_A = 0
                                        r_B = 0
                                        r_A_1 = 0
                                        r_B_1 = 0
                                        done = False
                                        done_1 = False
                                        action = self.encode_action(a_A, a_B)
                                        #print(a_A, a_B)


                                        if a_A == 0:
                                            pass

                                        elif a_A == 1:
                                            new_row_A = max(row_A-1, 0)
                                            
                                        elif a_A == 2: #south
                                            new_row_A = min(row_A+1, maxR)
                                            
                                        elif a_A == 3: # east
                                            new_col_A = min(col_A+1, maxC)

                                        elif a_A == 4: # west
                                            new_col_A = max(col_A-1, 0)

                                        if a_B == 0: #stick
                                                pass
                                            
                                        elif a_B == 1: #north
                                            new_row_B = max(row_B-1, 0)

                                        elif a_B == 2: # south
                                            new_row_B = min(row_B+1, maxR)

                                        elif a_B == 3: #east
                                            new_col_B = min(col_B+1, maxC)

                                        elif a_B == 4: #west
                                            new_col_B = max(col_B-1, 0)

                                        new_col_A_1, new_col_B_1, new_row_A_1, new_row_B_1, r_A_1, r_B_1, new_ball_1 = new_col_A, new_col_B, new_row_A, new_row_B, r_A, r_B, new_ball

                                        #when A goes first
                                        if new_row_A == row_B and new_col_A == col_B: 
                                            new_row_A = row_A
                                            new_col_A = col_A
                                            if ball == -1: new_ball = 1

                                        if new_row_B == new_row_A and new_col_B == new_col_A:
                                            new_row_B = row_B
                                            new_col_B = col_B
                                            if ball == 1: new_ball = -1

                                        if (new_ball == -1 and new_col_A == 0) or (new_ball == 1 and new_col_B == 0):
                                            done = True
                                            r_A = 100
                                            r_B = -100

                                        if (new_ball == 1 and new_col_B == 3) or (new_ball == -1 and new_col_A == 3):
                                            done = True
                                            r_A = -100
                                            r_B = 100

                                        #when B goes first
                                        if new_row_B_1 == row_A and new_col_B_1 == col_A: 
                                            new_row_B_1 = row_B
                                            new_col_B_1 = col_B
                                            if ball == 1: new_ball_1 = -1

                                        if new_row_A_1 == new_row_B_1 and new_col_A_1 == new_col_B_1:
                                            new_row_A_1 = row_A
                                            new_col_A_1 = col_A
                                            if ball == -1: new_ball_1 = 1

                                        if (new_ball_1 == -1 and new_col_A_1 == 0) or (new_ball_1 == 1 and new_col_B_1 == 0):
                                            done_1 = True
                                            r_A_1 = 100
                                            r_B_1 = -100

                                        if (new_ball_1 == 1 and new_col_B_1 == 3) or (new_ball_1 == -1 and new_col_A_1 == 3):
                                            done_1 = True
                                            r_A_1 = -100
                                            r_B_1 = 100

                                        new_state = self.encode_state(new_col_A, new_col_B, new_row_A, new_row_B, new_ball)
                                        new_state_1 = self.encode_state(new_col_A_1, new_col_B_1, new_row_A_1, new_row_B_1, new_ball_1)
                                        #print(state,action)
                                        #print(new_state, r_A, r_B, done)
                                        P[state][action] += [(0.5, new_state, r_A, r_B, done), (0.5, new_state_1, r_A_1, r_B_1, done_1)]

        self.P = P
        self.reset()

        pass

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self,):
        s = self.encode_state(col_A=2, col_B=1, row_A=0, row_B=0, ball=1)
        self.s = s
        return s

    def step_decoded_action(self, a_A, a_B): #take separate actions from A and B
        action = self.encode_action(a_A, a_B)
        transitions = self.P[self.s][action]
        #print([t for t in transitions])
        i = categorical_sample([t[0] for t in transitions])
        #print(transition)
        p, s, r_A, r_B, done = transitions[i]
        self.s = s
        self.lastaction = action
        return (s, r_A, r_B, done, {"prob": p})

    def step_encoded_action(self, action): #take encoded joint action
        transitions = self.P[self.s][action]
        #print([t for t in transitions])
        i = categorical_sample([t[0] for t in transitions])
        #print(transition)
        p, s, r_A, r_B, done = transitions[i]
        self.s = s
        self.lastaction = action
        return (s, r_A, r_B, done, {"prob": p})

    def step_A(a_A):
        step(a_A, 0)
        return self.s

    def step_B(a_B):
        step(0, a_B)
        return self.s

    def encode_state(self, col_A, col_B, row_A, row_B, ball): #A has ball --> ball=-1, B has ball --> ball=1
        # 4, 4, 2, 2, 2
        i = col_A
        i *= 4
        i += col_B
        i *= 2
        i += row_A
        i *= 2
        i += row_B
        i *= 2
        if ball == -1: ball = 0
        i += ball
        return i

    def decode_state(self, i):
        out = []
        out.append(i % 2)
        i = i // 2
        out.append(i % 2)
        i = i // 2
        out.append(i % 2)
        i = i // 2
        out.append(i % 4)
        i = i // 4
        out.append(i)
        assert 0 <= i < 4
        out = list(reversed(out))
        if out[-1] == 0: out[-1] = -1
        return out

    def encode_action(self, a_A, a_B): #A has ball --> ball=-1, B has ball --> ball=1
        # 5, 5
        i = a_A
        i *= 5
        i += a_B
        return i

    def decode_action(self, i):
        out = []
        out.append(i % 5)
        i = i // 5
        out.append(i)        
        assert 0 <= i < 5
        out = list(reversed(out))
        return out

    def render(self, ):
        field = [[" ", " ", " ", " "],
                [" ", " ", " ", " "]]

        col_A, col_B, row_A, row_B, ball = self.decode_state(self.s)
        if ball == 1:
            field[row_A][col_A] = "a"
            field[row_B][col_B] = "B" #captialized means in possession of the ball
        elif ball == -1:
            field[row_A][col_A] = "A"
            field[row_B][col_B] = "b"

        line1 = " | ".join(e for e in field[0])
        line2 = " | ".join(e for e in field[1])

        out = "-----------------\n| {} |\n-----------------\n| {} |\n-----------------".format(line1, line2)
        print(out)




if __name__ == '__main__':

    print("this is the code for the soccer environment")

    a = soccer()
    #print(a.encode_state(3,3,1,1,-1))
    #print(a.decode_state(126))
    #print(a.encode_action(1,1))
    #print(a.decode_action(24))
    print(a.s)
    print(a.decode_state(a.s))
    #print(a.P[73][a.encode_action(2,0)])
    a.render()
    s, r_A, r_B, done, p = a.step_decoded_action(0,3)
    print(s, r_A, r_B, done, p)
    print(a.decode_state(s))
    a.render()
    s, r_A, r_B, done, p = a.step_decoded_action(2,0)
    print(s, r_A, r_B, done, p)
    print(a.decode_state(s))
    a.render()
    s, r_A, r_B, done, p = a.step_decoded_action(4,0)
    print(s, r_A, r_B, done, p)
    print(a.decode_state(s))
    a.render()
    s, r_A, r_B, done, p = a.step_decoded_action(4,0)
    print(s, r_A, r_B, done, p)
    print(a.decode_state(s))
    a.render()


    

    

    