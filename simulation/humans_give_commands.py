import numpy as np
def humans_give_commands(CTRL, ACM, t):
    """ Console @ CL_TS """

    if t < 0.3:
        CTRL.cmd_rpm = 100
        ACM.TLoad = 0.3
    elif t < 0.8:
        CTRL.cmd_rpm = 150
        ACM.TLoad = 0.7
        CTRL.cmd_idq[1] = 1
        # CTRL.index_voltage_model_flux_estimation = 4
    elif t < 1.5:
        CTRL.cmd_rpm = 150
        ACM.TLoad = 1
        CTRL.cmd_idq[1] = 3

    if CTRL.bool_overwrite_speed_commands == False:
        if t < 1.0:
            CTRL.cmd_rpm = 50
        elif t < 1.5:
            ACM.TLoad = 2
        elif t < 2.0:
            CTRL.cmd_rpm = 200
        elif t < 3.0:
            CTRL.cmd_rpm = -200
        elif t < 4.0:
            CTRL.cmd_rpm = 0
        elif t < 4.5:
            CTRL.cmd_rpm = 2000
        elif t < 5:
            CTRL.cmd_idq[0] = 2
        elif t < 5.5:
            ACM.TLoad = 0.0
        elif t < 6: 
            CTRL.CMD_SPEED_SINE_RPM = 500
        # else: # don't implement else to receive commands from IPython console

        # if CTRL.CMD_SPEED_SINE_RPM!=0:
        #     CTRL.cmd_rpm = CTRL.CMD_SPEED_SINE_RPM * np.sin(2*np.pi*CTRL.CMD_SPEED_SINE_HZ*t)
        pass

    if CTRL.bool_apply_sweeping_frequency_excitation == True:

        if CTRL.timebase > CTRL.CMD_SPEED_SINE_END_TIME:
            # next frequency
            CTRL.CMD_SPEED_SINE_HZ += CTRL.CMD_SPEED_SINE_STEP_SIZE
            # next end time
            CTRL.CMD_SPEED_SINE_LAST_END_TIME = CTRL.CMD_SPEED_SINE_END_TIME
            CTRL.CMD_SPEED_SINE_END_TIME += 1.0/CTRL.CMD_SPEED_SINE_HZ # 1.0 Duration for each frequency

        if CTRL.CMD_SPEED_SINE_HZ > CTRL.CMD_SPEED_SINE_HZ_CEILING:
            # stop
            CTRL.cmd_rpm = 0.0
            CTRL.cmd_idq[1] = 0.0
        else:
            # speed control - closed-loop sweep
            CTRL.cmd_rpm    = CTRL.CMD_SPEED_SINE_RPM      * np.sin(2*np.pi*CTRL.CMD_SPEED_SINE_HZ*(CTRL.timebase - CTRL.CMD_SPEED_SINE_LAST_END_TIME))

            # speed control - open-loop sweep
            CTRL.cmd_idq[1] = CTRL.CMD_CURRENT_SINE_AMPERE * np.sin(2*np.pi*CTRL.CMD_SPEED_SINE_HZ*(CTRL.timebase - CTRL.CMD_SPEED_SINE_LAST_END_TIME))


