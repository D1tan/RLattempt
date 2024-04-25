import cv2
import gym
import time
import numpy as np
import pywintypes
import win32gui, win32ui, win32con, win32api
from gym import spaces
import pydirectinput
import pytesseract
import directkeys                              # Pytesseract is not just a simple pip install.
from EldenReward import EldenReward
from walkToBoss import walkToBoss



#N_CHANNELS = 1                                  #Image format
IMG_WIDTH = 1021                                #Game capture resolution
IMG_HEIGHT = 573                             
MODEL_WIDTH = int(1021/2)                  #Ai vision resolution
MODEL_HEIGHT = int(573/2)


'''Ai action list'''
DISCRETE_ACTIONS = {'release_wasd': 'release_wasd',
                    'w': 'run_forwards',                
                    's': 'run_backwards',
                    'a': 'run_left',
                    'd': 'run_right',
                    'w+shift': 'dodge_forwards',
                    's+shift': 'dodge_backwards',
                    'a+shift': 'dodge_left',
                    'd+shift': 'dodge_right',
                    'c': 'attack',
                    'v': 'strong_attack',
                    'q+v': 'weapon_art_heavy',


                    'w+shift+space+c' : 'jump_attack',
                    'w+shift+space+v' : 'jump_strong_attack',

                    'e': 'use_item'}

NUMBER_DISCRETE_ACTIONS = len(DISCRETE_ACTIONS)
NUM_ACTION_HISTORY = 10                         #Number of actions the agent can remember


class EldenEnv(gym.Env):
    """Custom Elden Ring Environment that follows gym interface"""


    def __init__(self, config):
        '''Setting up the environment'''
        super(EldenEnv, self).__init__()

        '''Setting up the gym spaces'''
        self.action_space = spaces.Discrete(NUMBER_DISCRETE_ACTIONS)                                                            #Discrete action space with NUM_ACTION_HISTORY actions to choose from
        spaces_dict = {                                                                                                         #Observation space (img, prev_actions, state)
            'img': spaces.Box(low=0, high=255, shape=(510, 286, 4), dtype=np.uint8),                      #Image of the game
            'prev_actions': spaces.Box(low=0, high=1, shape=(NUM_ACTION_HISTORY, NUMBER_DISCRETE_ACTIONS, 1), dtype=np.uint8),      #Last 10 actions as one hot encoded array
            'state': spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32),                                                       #Stamina and helth of the player in percent
        }
        self.observation_space = gym.spaces.Dict(spaces_dict)
    

        '''Setting up the variables'''''
        pytesseract.pytesseract.tesseract_cmd = config["PYTESSERACT_PATH"]          #Setting the path to pytesseract.exe            
        self.reward = 0                                                             #Reward of the previous step
        self.rewardGen = EldenReward(config)                                        #Setting up the reward generator class
        self.death = False                                                          #If the agent died
        self.duel_won = False                                                       #If the agent won the duel
        self.t_start = time.time()                                                  #Time when the training started
        self.done = False                                                           #If the game is done
        self.step_iteration = 0                                                     #Current iteration (number of steps taken in this fight)
        self.first_step = True                                                      #If this is the first step
        self.max_reward = None                                                      #The maximum reward that the agent has gotten in this fight
        self.reward_history = []                                                    #Array of the rewards to calculate the average reward of fight
        self.action_history = []                                                    #Array of the actions that the agent took.
        self.time_since_heal = time.time()                                          #Time since the last heal
        self.action_name = ''                                                       #Name of the action for logging
        self.MONITOR = config["MONITOR"]                                            #Monitor to use
        self.DEBUG_MODE = config["DEBUG_MODE"]                                      #If we are in debug mode
        self.GAME_MODE = config["GAME_MODE"]                                        #If we are in PVP or PVE mode
        self.DESIRED_FPS = config["DESIRED_FPS"]                                    #Desired FPS (not implemented yet)
        self.BOSS_HAS_SECOND_PHASE = config["BOSS_HAS_SECOND_PHASE"]                #If the boss has a second phase
        self.are_in_second_phase = False
        self.last_frame = None  # Variable to store the last frame of the four frames
        self.observation_buffer = []  # Buffer to store observations
        self.observation_counter = 0                                     #If we are in the second phase of the boss
        if self.GAME_MODE == "PVE": self.walk_to_boss = walkToBoss(config["BOSS"])  #Class to walk to the boss
        else : 
            self.matchmaking = walkToBoss(99)                           #Matchmaking class for PVP mode
            self.duel_lockon = walkToBoss(100)                          #Lock on class to the player in PVP mode
            self.first_reset = True
    

    '''One hot encoding of the last 10 actions'''
    def oneHotPrevActions(self, actions):
        oneHot = np.zeros(shape=(NUM_ACTION_HISTORY, NUMBER_DISCRETE_ACTIONS, 1))
        for i in range(NUM_ACTION_HISTORY):
            if len(actions) >= (i + 1):
                oneHot[i][actions[-(i + 1)]][0] = 1
        #print(oneHot)
        return oneHot 


    '''Grabbing a screenshot of the game'''
    #def grab_screen_shot(self):
    #    monitor = self.sct.monitors[self.MONITOR]
    #    sct_img = self.sct.grab(monitor)
    #    frame = cv2.cvtColor(np.asarray(sct_img), cv2.COLOR_BGRA2BGR)
    #    frame = frame[31:604, 10:1031]    #cut the frame to the size of the game
    #    if self.DEBUG_MODE:
    #        self.render_frame(frame)
    #    return frame
    
    def grab_screen_shot(self):

        # 获取桌面
        hwin = win32gui.GetDesktopWindow()
        x = 10
        y = 31
        w = 1021
        h = 573

        # 返回句柄窗口的设备环境、覆盖整个窗口，包括非客户区，标题栏，菜单，边框
        hwindc = win32gui.GetWindowDC(hwin)

        # 创建设备描述表
        srcdc = win32ui.CreateDCFromHandle(hwindc)

        # 创建一个内存设备描述表
        memdc = srcdc.CreateCompatibleDC()

        # 创建位图对象
        bmp = win32ui.CreateBitmap()
        bmp.CreateCompatibleBitmap(srcdc, w, h)
        memdc.SelectObject(bmp)

        # 截图至内存设备描述表
        memdc.BitBlt((0, 0), (w, h), srcdc, (x, y), win32con.SRCCOPY)

        signedIntsArray = bmp.GetBitmapBits(True)
        img = np.fromstring(signedIntsArray, dtype='uint8')
        img.shape = (h, w, 4)

        # 内存释放
        srcdc.DeleteDC()
        memdc.DeleteDC()
        win32gui.ReleaseDC(hwin, hwindc)
        win32gui.DeleteObject(bmp.GetHandle())

        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    
    '''Rendering the frame for debugging'''
    def render_frame(self, frame):                
        cv2.imshow('debug-render', frame)
        cv2.waitKey(100)
        cv2.destroyAllWindows()
        
    
    '''Defining the actions that the agent can take'''
    def take_action(self, action):
        #action = -1 #Uncomment this for emergency block all actions
        if action == 0:
            directkeys.stop()
            self.action_name = 'stop'
        elif action == 1:
            directkeys.go_forward()
            self.action_name = 'w'
        #elif action == 2:
        #    pydirectinput.keyUp('w')
        #    pydirectinput.keyUp('s')
        #    pydirectinput.keyDown('s')
        #    self.action_name = 's'
        elif action == 2:
            directkeys.go_left()
            self.action_name = 'a'
        elif action == 3:
            directkeys.go_right()
            self.action_name = 'd'
        elif action == 4:
            directkeys.dodge_forward()
            self.action_name = 'dodge-forward'
        elif action == 5:
            directkeys.dodge_backward()
            self.action_name = 'dodge-backward'
        elif action == 6:
            directkeys.dodge_left()
            self.action_name = 'dodge-left'
        elif action == 7:
            directkeys.dodge_right()
            self.action_name = 'dodge-right'
        elif action == 8:
            directkeys.attack()
            self.action_name = 'attack'
        elif action == 9:
            directkeys.heavy()
            self.action_name = 'heavy'
        #elif action == 11:
        #    pydirectinput.press('x')
        #    self.action_name = 'magic'
        elif action == 10:                  #weapon art
            directkeys.weapon_art()
            self.action_name = 'weapon art'
        #elif action == 13:                  #weapon art heavy
        #    pydirectinput.press('q')
        #    pydirectinput.press('v')
        #    self.action_name = 'weapon art heavy'
        #    time.sleep(1.5)
        #elif action == 13:                  #running attack
        #    pydirectinput.keyDown('w')
        #    time.sleep(0.4)
        #    pydirectinput.keyDown('shift')
        #    time.sleep(0.35)
        #    pydirectinput.press('c')
        #    pydirectinput.keyUp('shift')
        #    self.action_name = 'running attack'
        #    time.sleep(0.2)
        #elif action == 14:                  #running heavy
        #    pydirectinput.keyDown('shift')
        #    pydirectinput.keyDown('w')
        #    time.sleep(0.35)
        #    pydirectinput.press('v')
        #    pydirectinput.keyUp('shift')
        #    self.action_name = 'running heavy'
        #    time.sleep(0.35)
        #elif action == 16:                  #running magic
        #    pydirectinput.keyDown('shift')
        #    pydirectinput.keyDown('w')
        #    time.sleep(0.35)
        #    pydirectinput.press('x')
        #    pydirectinput.keyUp('shift')
        #    self.action_name = 'running magic'
        elif action == 11:                  #jump attack
            #pydirectinput.keyDown('shift')
            directkeys.jump_attack()
            self.action_name = 'jump attack'
        elif action == 12:                  #jump heavy
            #pydirectinput.keyDown('shift')
            directkeys.jump_heavy()
            self.action_name = 'jump heavy'
        #elif action == 19:                  #jump magic
        #    pydirectinput.keyDown('shift')
        #    pydirectinput.keyDown('w')
        #    time.sleep(0.2)
        #    pydirectinput.press('space')
        #    time.sleep(0.1)
        #    pydirectinput.press('x')
        #    pydirectinput.keyUp('shift')
        #    self.action_name = 'jump magic' 
        #elif action == 18:                  #crouch attack
        #    pydirectinput.keyDown('w')
        #    time.sleep(0.2)  
        #    pydirectinput.press('x')
        #    time.sleep(0.2)
        #    pydirectinput.press('c')   
        #    pydirectinput.keyUp('w')
        #    self.action_name = 'crouch attack'  
        elif action == 13 and time.time() - self.time_since_heal > 1.5: #to prevent spamming heal we only allow it to be pressed every 1.5 seconds
            directkeys.heal()
            self.action_name = 'heal'
        elif action == 99:
            pydirectinput.press('esc')
            time.sleep(0.5)
            pydirectinput.press('right')
            time.sleep(0.4)
            
            pydirectinput.press('e')
            time.sleep(1.5)
            pydirectinput.press('left')
            time.sleep(0.5)
            pydirectinput.press('e')
            time.sleep(0.5)
            print('🔄🔥')
        
    
    '''Checking if we are in the boss second phase'''
    def check_for_second_phase(self):
        frame = self.grab_screen_shot()
        self.reward, self.death, self.boss_death, self.duel_won = self.rewardGen.update(frame, self.first_step) #perform eldenreward update to get the current status of the boss

        if not self.boss_death:                 #if the boss is not dead when we check for the second phase, we are in the second phase
            self.are_in_second_phase = True
        else:                                   #if the boss is dead we can simply warp back to the bonfire
            self.are_in_second_phase = False


    '''Waiting for the loading screen to end'''
    def wait_for_loading_screen(self):
        time_test=time.time()
        in_loading_screen = False           #If we are in a loading screen right now
        have_been_in_loading_screen = False #If a loading screen was detected
        t_check_frozen_start = time.time()  #Timer to check the length of the loading screen
        t_since_seen_next = None            #We detect the loading screen by reading the text "next" in the bottom left corner of the loading screen.
        while True: #We are forever taking a screenshot and checking if it is a loading screen.
            
            frame = self.grab_screen_shot()
            
            in_loading_screen = self.check_for_loading_screen(frame)
            
            if in_loading_screen:
                print("⌛ Loading Screen:", in_loading_screen) #Loading Screen: True
                have_been_in_loading_screen = True
                t_since_seen_next = time.time()
            else:   #If we dont see "next" on the screen we are not in the loading screen [anymore]
                if have_been_in_loading_screen:
                    print('⌛ After loading screen...')
                else:
                    print('⌛ Waiting for loading screen...')
                
            if have_been_in_loading_screen and (time.time() - t_since_seen_next) > 2.5:             #We have been in a loading screen and left it for more than 2.5 seconds
                print('⌛✔️ Left loading screen #1')
                break
            elif have_been_in_loading_screen and  ((time.time() - t_check_frozen_start) > 60):      #We have been in a loading screen for 60 seconds. We assume the game is frozen
                print('⌛❌ Did not leave loading screen #2 (Frozen)')
                #some sort of error handling here...
                #break
            elif not have_been_in_loading_screen and ((time.time() - t_check_frozen_start) > 20):   #We have not entered a loading screen for 25 seconds. (return to bonfire and walk to boss) #⚔️ in pvp we use this for waiting for matchmaking
                if self.GAME_MODE == "PVE":
                    if self.BOSS_HAS_SECOND_PHASE:
                        self.check_for_second_phase()
                        if self.are_in_second_phase:
                            print('⌛👹 Second phase found #3')
                            break
                        else:
                            print('⌛🔥 No loading screen found #3')
                            self.take_action(99)                #warp back to bonfire
                            t_check_frozen_start = time.time()  #reset the timer
                    else:
                        print('⌛🔥 No loading screen found #3')
                        self.take_action(99)                #warp back to bonfire
                        t_check_frozen_start = time.time()  #reset the timer
                                                            #try again by not breaking the loop (waiting for loading screen then walk to boss)
                else:
                    print('⌛❌ No loading screen found #3')
                    t_check_frozen_start = time.time()  #reset the timer
                                                        #continue waiting for loading screen (matchmaking)
            time.sleep(2) 
            

    '''Checking if we are in a loading screen'''
    def check_for_loading_screen(self, frame):
        next_text_image = frame[539:554, 80:108] #Cutting the frame to the location of the text "next" (bottom left corner)
        pytesseract_output = pytesseract.image_to_string(next_text_image,  lang='eng',config='--psm 7 --oem 3') #reading text from the image cut out
        in_loading_screen = "Next" in pytesseract_output or "next" in pytesseract_output             #Boolean if we see "next" in the text
        return in_loading_screen
        

    '''Step function that is called by train.py'''
    def step(self, action):
        #📍 Lets look at what step does
        #📍 1. Collect the current observation 
        #📍 2. Collect the reward based on the observation (reward of previous step)            #⚔️PvP reward
        #📍 3. Check if the game is done (player died, boss died, 10minute time limit reached)  #⚔️Or duel won
        #📍 4. Take the next action (based on the decision of the agent)
        #📍 5. Ending the step
        #📍 6. Returning the observation, the reward, if we are done, and the info
        #📍 7*. train.py decides the next action and calls step again


        if self.first_step: print("🐾#1 first step")
        
        '''Grabbing variables'''
        t_start = time.time()    #Start time of this step
        
        frame_reward = self.grab_screen_shot()                             #📍 1. Collect the current observation
        self.reward, self.death, self.boss_death, self.duel_won = self.rewardGen.update(frame_reward, self.first_step) #📍 2. Collect the reward based on the observation (reward of previous step)
        
        '''📍 3. Checking if the game is done'''
        if self.death:
            self.done = True
            with open("boss_hp_log.txt", "a") as file:
                file.write(f"Boss HP: {self.rewardGen.curr_boss_hp}, Time: {time.ctime()}\n")
            print('🐾✔️ Step done (player death)')
        else:
            if (time.time() - self.t_start) > 600:  #If the agent has been in control for more than 10 minutes we give up
                self.done = True
                self.take_action(99)                #warp back to bonfire
                print('🐾✔️ Step done (time limit)')
            elif self.boss_death:
                self.done = True   
                self.take_action(99)                #warp back to bonfire
                print('🐾✔️ Step done (boss death)')                                               
        '''📍 4. Taking the action'''
        
        if not self.done:
            self.take_action(action)
            time.sleep(0.1)
        

        '''📍 5. Ending the steap'''

        '''Return values'''
        while(self.observation_counter<4):
            frame = self.grab_screen_shot()
            frame = cv2.resize(frame, (MODEL_WIDTH, MODEL_HEIGHT))
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

            self.observation_buffer.append(frame)
            self.observation_counter += 1

            if self.observation_counter == 4:
                # Stack the frames to form the observation
                observation = np.stack(self.observation_buffer, axis=-1)  # This stacks along a new dimension
                # Reset for the next set of observations
                self.observation_buffer.clear()
        self.observation_counter = 0
        info = {}                                                       #Empty info for gym
        #observation = cv2.resize(frame, (MODEL_WIDTH, MODEL_HEIGHT))
        #observation =  cv2.cvtColor(observation,cv2.COLOR_BGR2GRAY)   #We resize the frame so the agent dosnt have to deal with a 1920x1080 image (400x225)              #🐜 If we are in debug mode we render the frame
        if self.max_reward is None:                                     #Max reward
            self.max_reward = self.reward
        elif self.max_reward < self.reward:
            self.max_reward = self.reward
        self.reward_history.append(self.reward)                         #Reward history
        spaces_dict = {                                                 #Combining the observations into one dictionary like gym wants it
            'img': observation,
            'prev_actions': self.oneHotPrevActions(self.action_history),
            'state': np.asarray([self.rewardGen.curr_hp, self.rewardGen.curr_stam])
        }


        '''Other variables that need to be updated'''
        self.first_step = False
        self.step_iteration += 1
        self.action_history.append(int(action))                         #Appending the action to the action history


        '''FPS LIMITER'''
        #t_end = time.time()                                             
        #desired_fps = (1/self.DESIRED_FPS)                            #My CPU (i9-13900k) can run the training at about 2.4SPS (steps per secons)
        #time_to_sleep = desired_fps - (t_end - t_start)
        #if time_to_sleep > 0:
        #    time.sleep(time_to_sleep)
        '''END FPS LIMITER'''


        current_fps = str(1)     #Current SPS (steps per second)


        '''Console output of the step'''
        if not self.done: #Losts of python string formatting to make the console output look nice
            self.reward = round(self.reward, 0)
            reward_with_spaces = str(self.reward)
            for i in range(5 - len(reward_with_spaces)):
                reward_with_spaces = ' ' + reward_with_spaces
            max_reward_with_spaces = str(self.max_reward)
            for i in range(5 - len(max_reward_with_spaces)):
                max_reward_with_spaces = ' ' + max_reward_with_spaces
            for i in range(18 - len(str(self.action_name))):
                self.action_name = ' ' + self.action_name
            for i in range(5 - len(current_fps)):
                current_fps = ' ' + current_fps
            print('👣 Iteration: ' + str(self.step_iteration) + '| FPS: ' + current_fps + '| Reward: ' + reward_with_spaces + '| Max Reward: ' + max_reward_with_spaces + '| Action: ' + str(self.action_name))
        else:           #If the game is done (Logging Reward for dying or winning)
            print('👣✔️ Reward: ' + str(self.reward) + '| Max Reward: ' + str(self.max_reward))


        #📍 6. Returning the observation, the reward, if we are done, and the info
        return spaces_dict, self.reward, self.done, info
    

    '''Reset function that is called if the game is done'''
    def reset(self):
        #📍 1. Clear any held down keys
        #📍 2. Print the average reward for the last run
        #📍 3. Wait for loading screen                      #⚔️3-4 PvP: wait for loading screen - matchmaking - wait for loading screen - lock on
        #📍 4. Walking back to the boss
        #📍 5. Reset all variables
        #📍 6. Create the first observation for the first step and return it


        print('🔄 Reset called...')


        '''📍 1.Clear any held down keys'''
        self.take_action(0)
        print('🔄🔪 Unholding keys...')

        '''📍 2. Print the average reward for the last run'''
        if len(self.reward_history) > 0:
            total_r = 0
            for r in self.reward_history:
                total_r += r
            avg_r = total_r / len(self.reward_history)                              
            print('🔄🎁 Average reward for last run:', avg_r)
            print('🔄🎁 Total reward for last run:', total_r) 
            
            


        '''📍 3. Checking for loading screen / waiting some time for sucessful reset'''
        if self.GAME_MODE == "PVE": self.wait_for_loading_screen()
        
            

        '''📍 4. Walking to the boss'''         #⚔️we already did this in 📍 3. for PVP
        if self.GAME_MODE == "PVE":
            if self.BOSS_HAS_SECOND_PHASE:
                if self.are_in_second_phase:
                    print("🔄👹 already in arena")
                else:
                    print("🔄👹 walking to boss")
                    self.walk_to_boss.perform()
            else:                
                print("🔄👹 walking to boss")
                self.walk_to_boss.perform()          #This is hard coded in walkToBoss.py

        #if self.death:                           #Death counter in txt file
        #    f = open("deathCounter.txt", "r")
        #    deathCounter = int(f.read())
        #    f.close()
        #    deathCounter += 1
        #    f = open("deathCounter.txt", "w")
        #    f.write(str(deathCounter))
        #    f.close()


        '''📍 5. Reset all variables'''
        self.step_iteration = 0
        self.reward_history = [] 
        self.done = False
        self.first_step = True
        self.max_reward = None
        self.rewardGen.prev_hp = 1
        self.rewardGen.curr_hp = 1
        self.rewardGen.time_since_dmg_taken = time.time()
        self.rewardGen.curr_boss_hp = 1
        self.rewardGen.prev_boss_hp = 1
        self.action_history = []
        self.t_start = time.time()


        '''📍 6. Return the first observation'''
        while(self.observation_counter<4):
            frame = self.grab_screen_shot()
            frame = cv2.resize(frame, (MODEL_WIDTH, MODEL_HEIGHT))
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

            self.observation_buffer.append(frame)
            self.observation_counter += 1

            if self.observation_counter == 4:
                # Stack the frames to form the observation
                observation = np.stack(self.observation_buffer, axis=-1)  # This stacks along a new dimension
                # Reset for the next set of observations
                self.observation_buffer.clear()
        self.observation_counter = 0
        spaces_dict = { 
            'img': observation,                                         #The image
            'prev_actions': self.oneHotPrevActions(self.action_history),#The last 10 actions (empty)
            'state': np.asarray([1.0, 1.0])                             #Full hp and full stamina
        }
        
        print('🔄✔️ Reset done.')
        return spaces_dict                                              #return the new observation
