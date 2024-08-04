# %%
from rich import print

DISABLE_STUPID_WARNING = True


# data frame example
class CustomDataFrame:
    def __init__(self) -> None:
        self.plot_details = []

    def load(self, path1, path2):
        with open(path1, 'r', encoding='utf-8') as f:
            user_figs = f.read()
        with open(path2, 'r', encoding='utf-8') as f:
            signals_library = f.read()
        user_figs = user_figs.split('\n')
        signals_library = signals_library.split('\n')
        try:
            for i in range(len(user_figs)):
                user_fig = user_figs[i].split('@')
                for j in range(len(user_fig)):
                    user_fig[j] = user_fig[j].strip()
                if user_fig[0] == '':
                    continue
                self.plot_details.append({'data_axis': user_fig[0],
                                          'data_signal': user_fig[1:],
                                          'data_signal_label': user_fig[1:],
                                          'data_signal_num': len(user_fig[1:])})
            for i in range(len(signals_library)):
                signal = signals_library[i].split('@')
                for j in range(len(signal)):
                    signal[j] = signal[j].strip()
                if signal[0] == '':
                    continue
                for k in range(len(self.plot_details)):
                    for l in range(len(self.plot_details[k]['data_signal'])):
                        if signal[0] == self.plot_details[k]['data_signal'][l]:
                            self.plot_details[k]['data_signal_label'][l] = signal[3]
        except:
            raise Exception('user_cjh.txt or signals_library is not in the correct format.')

    def generate_function(self):
        with open(os.path.dirname(__file__) + '/collect_data.py', 'w') as f:
            f.write(
                f'''import numpy as np\ndef collect_data(watch_data, watch_index, CTRL, ACM, reg_id, reg_iq, reg_speed, fe_htz, fe_syn, fe_no_sat):\n''')
            index = 0
            for i in range(len(self.plot_details)):
                for j in range(len(self.plot_details[i]['data_signal'])):
                    f.write(f'\twatch_data[{index}][watch_index] = {self.plot_details[i]["data_signal"][j]}\n')
                    index += 1
            f.write(f'''\twatch_index += 1\n\treturn watch_index''')

   
    def plot(self, machine_times, watch_data, Lq_param=1.0, Rs_param=1.0, ELL_param = 0.1):
        plt.style.use('bmh')  # https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html
        mpl.rc('font', family='Times New Roman', size=8)
        mpl.rc('legend', fontsize=8)
        mpl.rcParams['lines.linewidth'] = 1  # mpl.rc('lines', linewidth=4, linestyle='-.')
        mpl.rc('font', family='Times New Roman', size=8)
        mpl.rc('legend', fontsize=8)
        mpl.rcParams['lines.linewidth'] = 1  # mpl.rc('lines', linewidth=4, linestyle='-.')
        mpl.rcParams['mathtext.fontset'] = 'stix'

        total = len(self.plot_details)
        index = 0
        figure_index = 0
        result = {}
        result = {}

        if total < 6:
            fig, axes = plt.subplots(nrows=total, ncols=1, dpi=150, facecolor='w', figsize=(8, 12), sharex=True)
        else:
            fig, axes_ = plt.subplots(nrows=(total + 1) // 2, ncols=2, dpi=150, facecolor='w', figsize=(8, 8),
                                      sharex=True)
            axes = np.ravel(axes_)
        linecolor = ['black', 'orange', 'olive', 'red','green', 'purple', 'brown', 'pink', 'gray', 'cyan']
        difflinestyle = ['-', '--', '-.', ':']
        for plot_detail in self.plot_details:
            ax = axes[figure_index]
            figure_index += 1
            for i in range(len(plot_detail['data_signal'])):
                ax.plot(machine_times, watch_data[index], label=plot_detail['data_signal_label'][i],linestyle = difflinestyle[i] , linewidth=1, color=linecolor[i], alpha=0.7)
                result[plot_detail['data_signal'][i]] = watch_data[index]
                result[plot_detail['data_signal'][i]] = watch_data[index]
                index += 1
            ax.set_ylabel(plot_detail['data_axis'], multialignment='center', color ='black')
            ax.legend(loc=1, fontsize=12, framealpha=0.38)
            ax.grid(True)
            ax.tick_params(colors='black') 
            for spine in ax.spines.values():  # Add these lines

                spine.set_edgecolor('black')

                spine.set_linewidth(0.5) 

        axes[-1].set_xlabel('Time [s]', color ='black')

        plt.show()
        return result


    def lissajou(self, watch_data_as_dict, period, path, Lq_param=1.0, Rs_param=1.0, ELL_param = 0.1):
        with open(path, 'r', encoding='utf-8') as f:
            user_figs = f.read()
        user_figs = user_figs.split('\n')
        user_fig_config = []
        try:
            for i in range(len(user_figs)):
                user_fig = user_figs[i].split('@')
                for j in range(len(user_fig)):
                    user_fig[j] = user_fig[j].strip()
                if user_fig[0]:
                    user_fig_config.append(user_fig)
        except:
            raise Exception('user_yzz.txt is not in the correct format.')
        plt.style.use('bmh')  # https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html
        mpl.rc('font', family='Times New Roman', size=16.0)
        mpl.rc('legend', fontsize=16)
        mpl.rcParams['lines.linewidth'] = 3  # mpl.rc('lines', linewidth=4, linestyle='-.')
        mpl.rcParams['mathtext.fontset'] = 'stix'


        total = len(user_fig_config)
        figure_index = 0
        axes = None
        axes = None

        if total < 6:
            fig, axes = plt.subplots(nrows=total, ncols=1, dpi=150, facecolor='w', figsize=(8, 12))
        else:
            fig, axes_ = plt.subplots(nrows=(total + 1) // 2, ncols=2, dpi=150, facecolor='w', figsize=(8, 12))
            axes = np.ravel(axes_)

        for i in range(len(user_fig_config)):
            ax = axes[figure_index]
            figure_index += 1
            x_lim_low = min(watch_data_as_dict[user_fig_config[i][0]][int(float(user_fig_config[i][4])/period): int(float(user_fig_config[i][5])/period)])
            x_lim_high = max(watch_data_as_dict[user_fig_config[i][0]][int(float(user_fig_config[i][4])/period): int(float(user_fig_config[i][5])/period)])
            x_lim_low = min(watch_data_as_dict[user_fig_config[i][0]][int(float(user_fig_config[i][4])/period): int(float(user_fig_config[i][5])/period)])
            x_lim_high = max(watch_data_as_dict[user_fig_config[i][0]][int(float(user_fig_config[i][4])/period): int(float(user_fig_config[i][5])/period)])
            x_lim_shift = (x_lim_high - x_lim_low) / 10
            ax.plot(watch_data_as_dict[user_fig_config[i][0]][int(float(user_fig_config[i][4])/period): int(float(user_fig_config[i][5])/period)],
                    watch_data_as_dict[user_fig_config[i][1]][int(float(user_fig_config[i][4])/period): int(float(user_fig_config[i][5])/period)], 
                    color='#8FBC8F')
                    #, '.')
            ax.set_xlabel(user_fig_config[i][2], multialignment='center')
            ax.set_ylabel(user_fig_config[i][3], multialignment='center')
            ax.set_xlim(x_lim_low - x_lim_shift, x_lim_high + x_lim_shift)
            ax.set_ylim(x_lim_low - x_lim_shift, x_lim_high + x_lim_shift)
            ax.grid(True)
            ax.set_aspect(aspect='equal')
        return None


custom = CustomDataFrame()
import os
custom.load(os.path.dirname(__file__) + '/user_cjh.txt', 
            os.path.dirname(__file__) + '/signals_library.txt')
custom.generate_function()

d = {
    # Timing
    'CL_TS': 1e-4,
    'TIME_SLICE': 12,
    'NUMBER_OF_SLICES': 1,
    'VL_EXE_PER_CL_EXE': 5,
    'MACHINE_SIMULATIONs_PER_SAMPLING_PERIOD': 1,
    'FOC_delta': 6.5,  # 25, # 6.5
    'FOC_desired_VLBW_HZ': 50,  # 60
    'FOC_CL_KI_factor_when__bool_apply_decoupling_voltages_to_current_regulation__is_False': 10,
    'CL_SERIES_KP': None,
    'CL_SERIES_KI': None,
    'VL_SERIES_KP': None,
    'VL_SERIES_KI': None,
    'VL_LIMIT_OVERLOAD_FACTOR': 3.0,
    'disp.Kp': 0.0,
    'disp.Ki': 0.0,
    'disp.Kd': 0.0,
    'disp.tau': 0.0,
    'disp.OutLimit': 0.0,
    'disp.IntLimit': 0.0,
    #Control index
    'CTRL.index_voltage_model_flux_estimation': 5,
    'CTRL.index_separate_speed_estimation': 0,
    'CTRL.bool_apply_decoupling_voltages_to_current_regulation':  False,
    'CTRL.bool_apply_speed_closed_loop_control': True,
    'CTRL.bool_zero_id_control': True,
    'CTRL.bool_reverse_rotation': False,
    'CTRL.bool_overwrite_speed_commands': True, #False才运行
    'CTRL.bool_apply_sweeping_frequency_excitation': False,
    'CTRL.use_encoder_angle_no_matter_what': True,
    'CTRL.index_controller': 0,
    'bool_rs_est_on': False,
    'OFFSET_VOLTAGE_ALPHA':0.0,
    'OFFSET_VOLTAGE_BETA':0.0,
    'NO_Saturation_PI_CORRECTION_GAIN_P':8,
    'NO_Saturation_PI_CORRECTION_GAIN_I':7,
}
# According to CTRL.index_separate_speed_estimation, chose your speed observer
#0. encoder for speed
#1. SEPARATE_SPEED_OBSERVER
#2. NATRUE_SPEED_OBSERVER
#3. marino_2005

# 小电感电机
# # 舞肌电机
# d['init_npp'] = 26
# d['init_IN'] = 17 
# d['init_R'] = 0.12
# d['init_Ld'] = 0.00046
# d['init_Lq'] = 0.00056
# d['init_KE'] = 0.019
# d['init_KA'] = 0.019
# d['init_Rreq'] = 0.0
# d['init_Js'] = 0.000364
# d['DC_BUS_VOLTAGE'] = 48
# MD 电机
d['init_npp'] = 5
d['init_IN'] = 16.8
d['init_R'] = 0.044
d['init_Ld'] = 0.00019
d['init_Lq'] = 0.00019
d['init_KE'] = 0.01717
d['init_KA'] = 0.01717
d['init_Rreq'] = 0.0
d['init_Js'] = 0.000755
d['DC_BUS_VOLTAGE'] = 48
# 北京时代超群
# d['init_npp'] = 4
# d['init_IN'] = 3
# d['init_R'] = 1.10
# d['init_Ld'] = 0.00496
# d['init_Lq'] = 0.00496
# d['init_KE'] = 0.1
# d['init_KA'] = 0.1
# d['init_Rreq'] = 0.0
# d['init_Js'] = 0.000617
# d['DC_BUS_VOLTAGE'] = 110


from tutorials_ep10_ParameterIdentification import *

print('Simulation_Benchmark')
# Auto-tuning PI
if d['CL_SERIES_KP'] is None:
    # sys.path.append(os.path.join(os.path.dirname(__file__), "tuner"))
    import tuner

    tuner.tunner_wrapper(d)
    print('\tAuto tuning...')
    print(f'\t{d=}\n')
else:
    print('\tSkip tuning.')

def InitialAllGlobalClass():
    
    CTRL = The_Motor_Controller(**d)
    ACM = The_AC_Machine(CTRL, MACHINE_SIMULATIONs_PER_SAMPLING_PERIOD=d['MACHINE_SIMULATIONs_PER_SAMPLING_PERIOD'])


    fe_htz = Variables_FluxEstimator_Holtz03(CTRL.R, d['init_KE'])

    fe_syn = Variables_FluxEstimator_Syn_IFO_lascu2006(CTRL.R, d['init_KE'])

    fe_no_sat = Variables_FluxEstimator_no_saturation_time_based(CTRL.R, d['init_KE'])

    reg_dispX = The_PID_Regulator(d['disp.Kp'], d['disp.Ki'], d['disp.Kd'], d['disp.tau'], d['disp.OutLimit'],
                                d['disp.IntLimit'], d['CL_TS'])
    reg_dispY = The_PID_Regulator(d['disp.Kp'], d['disp.Ki'], d['disp.Kd'], d['disp.tau'], d['disp.OutLimit'],
                                d['disp.IntLimit'], d['CL_TS'])

    if False:
        # Use incremental_pi codes
        reg_id = The_PI_Regulator(d['CL_SERIES_KP'], d['CL_SERIES_KP'] * d['CL_SERIES_KI'] * CTRL.CL_TS,
                                d['DC_BUS_VOLTAGE'] / 1.732)  # 我们假设调制方式是SVPWM，所以母线电压就是输出电压的线电压最大值，而我们用的是恒相幅值变换，所以限幅是相电压。
        reg_iq = The_PI_Regulator(d['CL_SERIES_KP'], d['CL_SERIES_KP'] * d['CL_SERIES_KI'] * CTRL.CL_TS,
                                d['DC_BUS_VOLTAGE'] / 1.732)  # 我们假设调制方式是SVPWM，所以母线电压就是输出电压的线电压最大值，而我们用的是恒相幅值变换，所以限幅是相电压。
        reg_speed = The_PI_Regulator(d['VL_SERIES_KP'], d['VL_SERIES_KP'] * d['VL_SERIES_KI'] * CTRL.VL_TS,
                                    d['VL_LIMIT_OVERLOAD_FACTOR'] * 1.414 * d['init_IN'])  # IN 是线电流有效值，我们这边限幅是用的电流幅值。
    else:
        # Use tustin_pi codes
        local_Kp = d['CL_SERIES_KP']
        if d['CTRL.bool_apply_decoupling_voltages_to_current_regulation'] == False:
            local_Ki = d['CL_SERIES_KP'] * d['CL_SERIES_KI'] * d[
                'FOC_CL_KI_factor_when__bool_apply_decoupling_voltages_to_current_regulation__is_False']
            print(
                '\tNote bool_apply_decoupling_voltages_to_current_regulation is False, to improve the current regulator performance a factor of %g has been multiplied to CL KI.' % (
                    d['FOC_CL_KI_factor_when__bool_apply_decoupling_voltages_to_current_regulation__is_False']))
        else:
            local_Ki = d['CL_SERIES_KP'] * d['CL_SERIES_KI']
        local_Kd = 0.0
        
        local_tau = 0.0
        local_OutLimit = d['DC_BUS_VOLTAGE'] / 1.732
        local_IntLimit = 1.0 * d[
            'DC_BUS_VOLTAGE'] / 1.732  # Integrator having a lower output limit makes no sense. For example, the q-axis current regulator needs to cancel back emf using the integrator output for almost full dc bus voltage at maximum speed.
        reg_id = The_PID_Regulator(local_Kp, local_Ki, local_Kd, local_tau, local_OutLimit, local_IntLimit, d['CL_TS'])
        reg_iq = The_PID_Regulator(local_Kp, local_Ki, local_Kd, local_tau, local_OutLimit, local_IntLimit, d['CL_TS'])
        print(f'\t{reg_id.OutLimit=} V')

        local_Kp = d['VL_SERIES_KP']
        local_Ki = d['VL_SERIES_KP'] * d['VL_SERIES_KI']
        local_Kd = 0.0
        local_tau = 0.0
        local_OutLimit = d['VL_LIMIT_OVERLOAD_FACTOR'] * 1.414 * d['init_IN']
        local_IntLimit = 1.0 * d['VL_LIMIT_OVERLOAD_FACTOR'] * 1.414 * d['init_IN']
        reg_speed = The_PID_Regulator(local_Kp, local_Ki, local_Kd, local_tau, local_OutLimit, local_IntLimit, CTRL.VL_TS)
        print(f'\t{reg_speed.OutLimit=} A')
    AllClass = (CTRL, ACM, reg_id, reg_iq, reg_speed, reg_dispX, reg_dispY, fe_htz, fe_syn, fe_no_sat)
    return  AllClass

ELL_param = [0.017]
Lq_param = [1]
P2PIndex = 0
for lq_param in Lq_param:
    rs_param = 1
    for ell_param in ELL_param:
        CTRL, ACM, reg_id, reg_iq, reg_speed, reg_dispX, reg_dispY, fe_htz, fe_syn, fe_no_sat = InitialAllGlobalClass()
        print(f'generate {lq_param} - {rs_param} - {ell_param}')
        d['Lq_param'] = lq_param
        d['Rs_param'] = rs_param
        d['ELL_param'] = ell_param
        # simulate to generate NUMBER_OF_SLICES*TIME_SLICE sec of data
        for ii in range(d['NUMBER_OF_SLICES']):
            # perform animation step
            CTRL = The_Motor_Controller(**d)
            ACM = The_AC_Machine(CTRL, MACHINE_SIMULATIONs_PER_SAMPLING_PERIOD=d['MACHINE_SIMULATIONs_PER_SAMPLING_PERIOD'],
                                Lq_param=d['Lq_param'])
            machine_times, watch_data = ACMSimPyIncremental(t0=ii * d['TIME_SLICE'], TIME=d['TIME_SLICE'],
                                                            ACM=ACM,
                                                            CTRL=CTRL,
                                                            reg_id=reg_id,
                                                            reg_iq=reg_iq,
                                                            reg_speed=reg_speed,
                                                            fe_htz=fe_htz,
                                                            fe_syn=fe_syn,
                                                            fe_no_sat=fe_no_sat,
                                                            Rs_param=d['Rs_param'],
                                                            ELL_param=d['ELL_param'])
            watch_data_as_dict = custom.plot(machine_times, watch_data, Lq_param=lq_param, Rs_param=rs_param, ELL_param = ell_param)
plt.show()
print("finish!")