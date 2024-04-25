import train


if __name__ == '__main__':
    '''User Settings'''
    env_config = {
        "PYTESSERACT_PATH": r'C:/eldenring/drl/tesseract/tesseract.exe',    # Set the path to PyTesseract
        "MONITOR": 1,           #Set the monitor to use (1,2,3)
        "DEBUG_MODE": False,    #Renders the AI vision (pretty scuffed)
        "GAME_MODE": "PVE",     #PVP or PVE
        "BOSS": 3,              #1-6 for PVE (look at walkToBoss.py for boss names) | Is ignored for GAME_MODE PVP
        "BOSS_HAS_SECOND_PHASE": False,  #Set to True if the boss has a second phase (only for PVE)
        "PLAYER_HP": 522,      #Set the player hp (used for hp bar detection)
        "PLAYER_STAMINA": 97,  #Set the player stamina (used for stamina bar detection)
        "DESIRED_FPS": 24    #Set the desired fps (used for actions per second) (24 = 2.4 actions per second) #not implemented yet       #My CPU (i9-13900k) can run the training at about 2.4SPS (steps per secons)
    }
    CREATE_NEW_MODEL=False #eeeeEECreate a new model or resume training for an existing model


    '''Start Training'''
    print("💍 EldenRL 💍")
    train.train(CREATE_NEW_MODEL, env_config)  