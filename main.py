import numpy as np
import yaml

class Bot:
  def __init__(self, params):

      
  def next_state(self, controls):
    new_config = self.config

    njoints = 5
    nwheels = 4

    dtheta_w = np.zeros(nwheels)
    dtheta_j = np.zeros(njoints)

    for i in range(nwheels):
      u = controls["wheel"][i]
      u = self.set_velocity(u, self.wheel_vel_limits[i])

      dtheta_w[i] = u * self.dt
      new_config["wheel"][i] += dtheta_w[i]

    for i in range(njoints):
      v = controls["arm"][i]
      v = self.set_velocity(v, self.joint_vel_limits[i])

      dtheta_j[i] = v * self.dt
      new_config["arm"][i] += dtheta_j[i]

    # Find the delta transformation matrix
    body_twist_V6 = np.dot(self.F6, dtheta_w)
    body_twist_se3 = mr.VecTose3(body_twist_V6)
    T_delta = mr.MatrixExp6(body_twist_se3)

    Tsb_new = np.dot(self.Tsb, T_delta)
    theta = math.atan2(Tsb_new[1,0], Tsb_new[0,0])

    new_config["chassis"][0] = theta
    new_config["chassis"][1] = Tsb_new[0,-1]
    new_config["chassis"][2] = Tsb_new[1,-1]

    return new_config

if __name__ == "__main__":
  paramsFile = "main.yaml"
  outputConfigFile = "output.csv"

  outputConfigs = list()

  with open(paramsFile) as file:
    botParams = yaml.load(file, Loader=yaml.FullLoader)

  bot = Bot(botParams)

  trajs = bot.generate_trajectory()

  # Tse_d   - Desired configuration
  # Tse_dn  - Desired next configuration
  time = 0.0
  bot.update_config(bot.config)

  bot.update_config_output_list(output_configs)

  for traj, gripper_state in trajs:
    for conf_no in range(len(traj)-1):
      Tse_d  = traj[conf_no]
      Tse_dn = traj[conf_no + 1]

      error_twist_v6, u, theta_dot = bot.feedback_control(Tse_d, Tse_dn)
      controls = {'arm' : list(theta_dot), 'wheel' : list(u)}
      new_config = bot.next_state(controls)
      new_config["gripper"] = gripper_state
      bot.update_config(new_config)

      bot.update_config_output_list(output_configs)
      bot.save_error(error_twist_v6)
      time = time + bot.dt

  print("Total simulation time: ", time)
  print("Generating output config file")
  with open(output_config_filename, 'w', newline='') as csv_file:
    writer = csv.writer(csv_file, delimiter=',')
    for i in output_configs:
      writer.writerow(i)

  bot.plot_error(time - bot.dt)     # subtracting the extra dt added to at the end.