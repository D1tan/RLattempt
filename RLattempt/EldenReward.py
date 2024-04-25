import cv2
import numpy as np
import time
import pytesseract 


class EldenReward:
    '''Reward Class'''


    '''Constructor'''
    def __init__(self, config):
        pytesseract.pytesseract.tesseract_cmd = config["PYTESSERACT_PATH"]        #Setting the path to pytesseract.exe
        self.GAME_MODE = config["GAME_MODE"]
        self.DEBUG_MODE = config["DEBUG_MODE"]
        self.max_hp = config["PLAYER_HP"]                             #This is the hp value of your character. We need this to capture the right length of the hp bar.
        self.prev_hp = 1.0     
        self.curr_hp = 1.0
        self.time_since_dmg_taken = time.time()
        self.death = False
        self.max_stam = config["PLAYER_STAMINA"]                     
        self.curr_stam = 1.0
        self.pre_boss_hp = 1.0
        self.curr_boss_hp = 1.0
        self.attack_multiplier = 1.0
        self.time_since_boss_dmg = time.time() 
        self.time_since_pvp_damaged = 0
        self.boss_hp_75 = False
        self.boss_hp_50 = False
        self.boss_hp_25 = False
        self.boss_hp_10 = False
        self.last_attack_time = time.time()
        self.boss_death = False    
        self.game_won = False    
        self.image_detection_tolerance = 0.02   #The image detection of the hp bar is not perfect. So we ignore changes smaller than this value. (0.02 = 2%)
        

    '''Detecting the current player hp'''
    def get_current_hp(self, frame):                                                        
        hp_image = frame[24:31, 80:392]     #Cut out the hp bar from the frame
        if self.DEBUG_MODE: self.render_frame(hp_image)
        
        image_hsv = cv2.cvtColor(hp_image, cv2.COLOR_BGR2HSV)
    
         # Define the color range for red in HSV
        lower_red1 = np.array([0, 70, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 70, 50])
        upper_red2 = np.array([180, 255, 255])
        
        # Create two masks for the two red ranges and combine them
        mask1 = cv2.inRange(image_hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(image_hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)
        
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Assume the largest contour is the health bar
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            health_bar_width = w
        else:
            # No red color detected, health bar is empty or an error occurred
            health_bar_width = 0
        
        # Calculate the percentage of the health bar
        curr_hp = round((health_bar_width / hp_image.shape[1]),2)
        if self.DEBUG_MODE: print('💊 Health: ', curr_hp)
        return curr_hp


    '''Detecting the current player stamina'''
    def get_current_stamina(self, frame):                                                        #Constant to calculate the length of the stamina bar
        stam_image = frame[44:50, 80:235]
        if self.DEBUG_MODE: self.render_frame(stam_image)

        lower = np.array([6,52,24])                                             #This filter really inst perfect but its good enough bcause stamina is not that important
        upper = np.array([74,255,77])                                           #Also Filter
        hsv = cv2.cvtColor(stam_image, cv2.COLOR_BGRA2BGR)                       #Apply the filter
        mask = cv2.inRange(hsv, lower, upper)                                   #Also apply
        if self.DEBUG_MODE: self.render_frame(mask)

        matches = np.argwhere(mask==255)                                        #Number for all the white pixels in the mask
        self.curr_stam = len(matches) / (stam_image.shape[1] * stam_image.shape[0]) #Calculating percent of white pixels in the mask (current stamina in percent)

        self.curr_stam += 0.02                                                  #Adding +2% of stamina for color noise
        if self.curr_stam >= 0.96:                                              #If the stamina is above 96% we set it to 100% (also color noise fix)
            self.curr_stam = 1.0 

        if self.DEBUG_MODE: print('🏃 Stamina: ', self.curr_stam)
        return self.curr_stam
    

    '''Detecting the current boss hp'''
    def get_boss_hp(self, frame):
        boss_hp_image = frame[461:468, 246:778]                             #cutting frame for boss hp bar (always same size)
        if self.DEBUG_MODE: self.render_frame(boss_hp_image)

        image_hsv = cv2.cvtColor(boss_hp_image, cv2.COLOR_BGR2HSV)
    
         # Define the color range for red in HSV
        lower_red1 = np.array([0, 70, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 70, 50])
        upper_red2 = np.array([180, 255, 255])
        
        # Create two masks for the two red ranges and combine them
        mask1 = cv2.inRange(image_hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(image_hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)
        
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Assume the largest contour is the health bar
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            health_bar_width = w
        else:
            # No red color detected, health bar is empty or an error occurred
            health_bar_width = 0
        
        # Calculate the percentage of the health bar
        boss_hp = round((health_bar_width / boss_hp_image.shape[1]),2)
        if self.DEBUG_MODE: self.render_frame(mask)


        #self.render_frame(boss_hp_image)
        #self.render_frame(mask)


        
        
        #same noise problem but the boss hp bar is larger so noise is less of a problem

        if self.DEBUG_MODE: print('👹 Boss HP: ', boss_hp)

        return boss_hp
    

    '''Detecting if the boss is damaged in PvE'''           #🚧 This is not implemented yet!!
    #def detect_boss_damaged(self, frame, former_hp):
    #    cut_frame = frame[461:468, 246:778]
    #    
    #    lower = np.array([23,210,0])                                            #This filter really inst perfect but its good enough bcause stamina is not that important
    #    upper = np.array([25,255,255])                                           #Also Filter
    #    hsv = cv2.cvtColor(cut_frame, cv2.COLOR_RGB2HSV)                       #Apply the filter
    #    mask = cv2.inRange(hsv, lower, upper)                                   #Also apply
    #    matches = np.argwhere(mask==255)                                        #Number for all the white pixels in the mask
    #    #self.render_frame(cut_frame)
    #    #self.render_frame(mask)
    #    if len(matches) < former_hp:                                                   #if there are more than 30 white pixels in the mask, return true
    #        return True
    #    else:
    #        return False

    

    '''Detecting if the enemy is damaged in PvP'''
    
    

    '''Debug function to render the frame'''
    

 
    '''Update function that gets called every step and returns the total reward and if the agent died or the boss died'''
    def update(self, frame, first_step, action_taken):
        #📍 1 Getting current values
        #📍 2 Hp Rewards
        #📍 3 Boss Rewards
        #📍 4 PvP Rewards
        #📍 5 Total Reward / Return

        
        '''📍1 Getting/Setting current values'''
        self.curr_hp = self.get_current_hp(frame)                   
        self.curr_stam = 1.0 #self.get_current_stamina(frame)            
        self.curr_boss_hp = self.get_boss_hp(frame)
        if first_step: self.time_since_dmg_taken = time.time() + 5 #Setting the time_since_dmg_taken to 10 seconds ago so we dont get a negative reward at the start of the game           

        self.death = False
        if self.curr_hp == 0:   #If our hp is below 1% we are dead
            self.death = True

        self.boss_death = False
        if self.GAME_MODE == "PVE":                                 #Only if we are in PVE mode
            if self.curr_boss_hp <= 0.1:                           #If the boss hp is below 1% the boss is dead (small tolerance because we want to be sure the boss is actually dead)
                self.boss_death = True

        
        '''📍2 Hp Rewards'''
        hp_reward = 0
        hp_change = self.curr_hp - self.prev_hp
        #if hp_change<-0.5:
        #        hp_reward = hp_change*2000
        if not self.death:                           
            if hp_change >  0.1:        #Reward if we healed)
                hp_reward = hp_change*300*(2-self.curr_hp)
                print("heal")                  
            if hp_change < -0.03 and hp_change>-0.5:      #Negative reward if we took damage
                hp_reward = hp_change*250*(2-self.curr_hp)      #可添加根据血量的多少时扣血的不同reward，血量越低，受击reward惩罚越高
                self.attack_multiplier = 1.0
                self.time_since_dmg_taken = time.time()
                print("hit")
            if hp_change < 0.05  and hp_change >-0.05 and (action_taken == 4 or action_taken == 5 or action_taken == 6 or action_taken == 7):
                hp_reward += 15
                print("dodge")
                
                
        #if self.death:
        #    hp_reward = hp_reward+int(200/(self.curr_boss_hp+0.1))-200                                                     #Large negative reward for dying

        time_since_taken_dmg_reward = 0                                  
        if time.time() - self.time_since_dmg_taken > 8:                             #Reward if we have not taken damage for 5 seconds (every step for as long as we dont take damage)
            time_since_taken_dmg_reward = 10

        print(f"PLAYER_HP={self.curr_hp}")
        self.prev_hp = self.curr_hp     #Update prev_hp to curr_hp


        '''📍3 Boss Rewards'''
        if self.GAME_MODE == "PVE":                                             #Only if we are in PVE mode
            boss_dmg_reward = 0
            boss_damage = self.pre_boss_hp-self.curr_boss_hp
            if self.curr_boss_hp < 0.03 and self.death:                         #error when bug appeared in the game                                       #Large reward if the boss is dead
                boss_dmg_reward = -200
            else:
                if boss_damage > 0.01:  #Reward if we damaged the boss (small tolerance because its a large bar)
                    if time.time() - self.last_attack_time > 5:
                        self.attack_multiplier = 1.0
                    boss_dmg_reward = boss_damage*1000*(2-self.curr_boss_hp)*self.attack_multiplier
                    self.attack_multiplier *= 1.2
                    self.last_attack_time = time.time()
                    print("damage")
                    self.time_since_boss_dmg = time.time()
                else:
                    if time.time() - self.last_attack_time > 5:
                        self.attack_multiplier = 1.0
                if time.time() - self.time_since_boss_dmg > 6:                      #Negative reward if we have not damaged the boss for 5 seconds (every step for as long as we dont damage the boss)
                    boss_dmg_reward = -10                                               
            if self.curr_boss_hp < 0.10 and not self.death and not self.boss_hp_10:
                boss_dmg_reward += 2000  # Major milestone reward
                self.boss_hp_10 = True
            elif self.curr_boss_hp < 0.25 and not self.death and not self.boss_hp_25:
                boss_dmg_reward += 1000
                self.boss_hp_25 = True
            elif self.curr_boss_hp < 0.50 and not self.death and not self.boss_hp_50:
                boss_dmg_reward += 500
                self.boss_hp_50 = True
            elif self.curr_boss_hp < 0.75 and not self.death and not self.boss_hp_75:
                boss_dmg_reward += 250
                self.boss_hp_75= True

            percent_through_fight_reward = 0
            #if self.curr_boss_hp < 0.97:                                            #Increasing reward for every step we are alive depending on how low the boss hp is
            #    percent_through_fight_reward = 5/self.curr_boss_hp            
            print(f"BOSS_HP={self.curr_boss_hp}")
            self.pre_boss_hp=self.curr_boss_hp
            
        '''📍5 Total Reward / Return'''
        if self.GAME_MODE == "PVE":                                                 #Only if we are in PVE mode
            total_reward = hp_reward + boss_dmg_reward + time_since_taken_dmg_reward + percent_through_fight_reward
        else:
            total_reward = hp_reward + time_since_taken_dmg_reward + pvp_reward
        
        total_reward = round(total_reward, 3)

        return total_reward, self.death, self.boss_death, self.game_won
    


#'''Testing code'''
#if __name__ == "__main__":
#    env_config = {
#        "PYTESSERACT_PATH": r'C:\Program Files\Tesseract-OCR\tesseract.exe',    # Set the path to PyTesseract
#        "MONITOR": 1,           #Set the monitor to use (1,2,3)
#        "DEBUG_MODE": False,    #Renders the AI vision (pretty scuffed)
#        "GAME_MODE": "PVE",     #PVP or PVE
#        "BOSS": 8,              #1-6 for PVE (look at walkToBoss.py for boss names) | Is ignored for GAME_MODE PVP
#        "BOSS_HAS_SECOND_PHASE": True,  #Set to True if the boss has a second phase (only for PVE)
#        "PLAYER_HP": 1679,      #Set the player hp (used for hp bar detection)
#        "PLAYER_STAMINA": 121,  #Set the player stamina (used for stamina bar detection)
#        "DESIRED_FPS": 24       #Set the desired fps (used for actions per second) (24 = 2.4 actions per second) #not implemented yet       #My CPU (i9-13900k) can run the training at about 2.4SPS (steps per secons)
#    }
#    reward = EldenReward(env_config)
#
#    IMG_WIDTH = 1920                                #Game capture resolution
#    IMG_HEIGHT = 1080  
#
#    import mss
#    sct = mss.mss()
#    monitor = sct.monitors[1]
#    sct_img = sct.grab(monitor)
#    frame = cv2.cvtColor(np.asarray(sct_img), cv2.COLOR_BGRA2RGB)
#    frame = frame[46:IMG_HEIGHT + 46, 12:IMG_WIDTH + 12]    #cut the frame to the size of the game
#
#    reward.update(frame, True)
#    time.sleep(1)
#    reward.update(frame, False)

        