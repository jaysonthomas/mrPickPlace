import mrlib as mr
import yaml
import numpy as np

from datetime import datetime, timedelta
import pandas as pd

from bokeh.io import output_notebook, show, save
from bokeh.layouts import row, grid
from bokeh.models import Span, Label, Range1d, Title, LinearAxis
from bokeh.models.sources import ColumnDataSource
from bokeh.models.tools import HoverTool
from bokeh.plotting import figure, show, output_file

import math
import csv

class Bot:
  def __init__(self, params):
    self.config = botParams["initialConfig"]
    # Trajectory
    self.Tse = botParams["Tse_initial"]
    self.Tce_so = botParams["Tce_standoff"]
    self.Tsc_i = botParams["Tsc_initial"]
    self.Tsc_f = botParams["Tsc_final"]
    self.Tce_grasp = botParams["Tce_grasp"]

    self.timeScaling = botParams["timeScaling"]
    self.trajType = botParams["trajType"]

    # Feedback control
    self.Tsb = np.zeros((4,4))
    self.thetalist = np.zeros(5)
    self.phi = 0
    self.x = 0
    self.y = 0
    self.Moe = np.array(botParams["arm"]["Moe"])
    self.Blist = np.array(botParams["arm"]["Blist"])
    self.Tbo = np.array(botParams["arm"]["Tbo"])
    self.dt = botParams["tanim"]
    self.k = botParams["k"]

    self.Kp = np.eye(6) * botParams["Kp"]
    self.Ki = np.eye(6) * botParams["Ki"]
    self.integral = np.zeros(6)

    self.l = botParams["chassis"]["l"]
    self.w = botParams["chassis"]["w"]
    self.rad = botParams["chassis"]["wheel_rad"]
    rval = 1/(self.l + self.w)

    self.F6 = np.array([[0,0,0,0],
                        [0,0,0,0],
                        [-rval, rval, rval, -rval],
                        [ 1, 1,  1, 1],
                        [-1, 1, -1, 1],
                        [0,0,0,0]])

    self.F6 = (self.rad/4) * self.F6

  def updateConfig(self, config):
    self.config = config
    self.thetalist = np.array(config["arm"])
    self.phi = config["chassis"][0]
    self.x = config["chassis"][1]
    self.y = config["chassis"][2]
    
    phi = self.phi
    self.Tsb = np.array([[math.cos(phi), -math.sin(phi), 0, self.x],
                         [math.sin(phi), math.cos(phi),  0, self.y],
                         [0, 0, 1, 0.0963],
                         [0, 0, 0, 1]])

# ------------------------------
# Feedback control
# ------------------------------
  def feedbackControl(self, Tse_d, Tse_dn):
    Toe = mr.FKinBody(self.Moe, self.Blist, self.thetalist)
    Tse_cur = np.dot(np.dot(self.Tsb, self.Tbo), Toe)

    Tse_cur_d = np.dot(mr.TransInv(Tse_cur), Tse_d)
    Tse_d_dn =  np.dot(mr.TransInv(Tse_d), Tse_dn)

    # Error term
    errorTwist_se3 = mr.MatrixLog6(Tse_cur_d)
    errorTwist_v6 = mr.se3ToVec(errorTwist_se3)

    # Feedforward term
    # Twist to go from the reference desired config to the next desired config
    ffTwist_se3 = mr.MatrixLog6(Tse_d_dn)/self.dt
    ffTwist_v6 = mr.se3ToVec(ffTwist_se3)
    
    # FF term = the end effector twist w.r.t. {s} to go from the current config to the next desired config.
    ff = np.dot(mr.Adjoint(Tse_cur_d), ffTwist_v6)

    # Proportional term
    p = np.dot(self.Kp, errorTwist_v6)

    # Integral term
    self.integral = self.integral + errorTwist_v6
    i = np.dot(self.Ki, self.integral * self.dt)

    # The end effector twist
    eeTwist = ff + p + i

    # To convert the end effector twist to the wheel/joint velocities, we need to find the jacobian
    Teb = np.dot(mr.TransInv(Toe), mr.TransInv(self.Tbo))
    J_u2ee = np.dot(mr.Adjoint(Teb), self.F6)

    J_theta2ee = mr.JacobianBody(self.Blist, self.thetalist)

    #Jacobian to map ee twist to wheel/joint velocities
    Jee = np.hstack((J_u2ee, J_theta2ee))
    
    # Jacobian mapping
    # TODO: tolerance of 1e-3 is added to pinv to avoid close to singularity situation
    vels = np.dot(np.linalg.pinv(Jee, 1e-3), eeTwist)
    
    return errorTwist_v6, vels[:4], vels[4:]   # eeTwist, wheel velocities (u), joint velocities (theta dot)

# ------------------------------
# Trajectory generation
# ------------------------------
  def getEulerAnglesFromRotationMatrix(self, R):
    # zyx
    def isclose(x, y, rtol=1.e-5, atol=1.e-8):
      return abs(x - y) <= atol + rtol * abs(y)

    phi = 0.0
    if isclose(R[2, 0], -1.0):
      theta = math.pi / 2.0
      psi = math.atan2(R[0, 1], R[0, 2])
    elif isclose(R[2, 0], 1.0):
      theta = -math.pi / 2.0
      psi = math.atan2(-R[0, 1], -R[0, 2])
    else:
      theta = -math.asin(R[2, 0])
      cos_theta = math.cos(theta)
      psi = math.atan2(R[2, 1] / cos_theta, R[2, 2] / cos_theta)
      phi = math.atan2(R[1, 0] / cos_theta, R[0, 0] / cos_theta)
    return psi, theta, phi

  def getSegmentDuration(self, start, end):
    # Units rad/s, m/s
    maxAngularVel = 0.1
    maxLinearVel = 0.1

    T_dist = np.dot(mr.TransInv(start), end)
    linearDist = math.hypot(T_dist[0, -1], T_dist[1, -1])
    rotationalDist = max(self.getEulerAnglesFromRotationMatrix(T_dist[:-1, :-1]))
    
    duration = max(linearDist/maxLinearVel, rotationalDist/maxAngularVel)
    return duration

  def getNumberOfConfigurationsInSegment(self, duration):
    nconf = int((duration * self.k) /self.dt)
    return nconf

  def generateTrajSegment(self, start, end):
    duration = self.getSegmentDuration(start, end)
    nconf = self.getNumberOfConfigurationsInSegment(duration)

    if (self.trajType == "screw"):
      trajs = mr.ScrewTrajectory(start, end, duration, nconf, self.timeScaling)
    elif (self.trajType == "cartesian"):
      trajs = mr.CartesianTrajectory(start, end, duration, nconf, self.timeScaling)

    return trajs

  def changeGripperState(self, Tse_current):
    trajs = list()

    for i in range(63):
      trajs.append(Tse_current)

    return trajs

  def generateReferenceTrajectory(self):
    trajs = list()
    GO = 0    # gripper open
    GC = 1    # gripper closed

    # Convert ee waypoint configurations w.r.t. {c} to w.r.t. {s}
    Tse_init_so     = np.dot(self.Tsc_i, self.Tce_so)
    Tse_init_grasp  = np.dot(self.Tsc_i, self.Tce_grasp)
    Tse_final_so    = np.dot(self.Tsc_f, self.Tce_so)
    Tse_final_grasp = np.dot(self.Tsc_f, self.Tce_grasp)

    # Each element of trajs is a list containing 2 elements.
    # Element 1: List of configurations. Each element of this list is a
    # numpy array. To access a particular configuration of the list: trajs[0][0][i]
    # Element 2: int - trajs[0][1]
    trajs.append([self.generateTrajSegment(self.Tse, Tse_init_so), GO])
    trajs.append([self.generateTrajSegment(Tse_init_so, Tse_init_grasp), GO])
    trajs.append([self.changeGripperState(Tse_init_grasp), GC])
    trajs.append([self.generateTrajSegment(Tse_init_grasp, Tse_init_so), GC])
    trajs.append([self.generateTrajSegment(Tse_init_so, Tse_final_so), GC])
    trajs.append([self.generateTrajSegment(Tse_final_so, Tse_final_grasp), GC])
    trajs.append([self.changeGripperState(Tse_final_grasp), GO])

    return trajs

# ------------------------------
# Simulation
# ------------------------------  
  def nextState(self, controls):
    new_config = self.config

    njoints = 5
    nwheels = 4

    dtheta_w = np.zeros(nwheels)
    dtheta_j = np.zeros(njoints)

    for i in range(nwheels):
      dtheta_w[i] = controls["wheel"][i] * self.dt
      new_config["wheel"][i] += dtheta_w[i]

    for i in range(njoints):
      dtheta_j[i] = controls["arm"][i] * self.dt
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

  def updateConfigOutputList(self, output):
    output.append([round(i, 4) for i in self.config["chassis"]] +
                  [round(i, 4) for i in self.config["arm"]] +
                  [round(i, 4) for i in self.config["wheel"]] +
                  [self.config["gripper"]])

def addHoverInfo(p, yvalue, xvalue):
  myTooltips = [
                ('index', '$index'),
                ('Y', yvalue),
                ('X', xvalue)
  ]
  hover = HoverTool(tooltips=myTooltips)
  hover.point_policy='snap_to_data'
  hover.line_policy='none'
  
  p.add_tools(hover)
  return p

def bokPlot(*p):
  if (len(p) == 1):
      l = grid([
          [p[0]]
      ], sizing_mode='stretch_both')
  elif (len(p) == 2):
      l = grid([
          [p[0]], 
          [p[1]]
      ], sizing_mode='stretch_both')
  elif (len(p) == 3):
      l = grid([
          [p[0]], 
          [p[1]],
          [p[2]]
      ], sizing_mode='stretch_both')
  elif (len(p) == 4):
      l = grid([
          [p[0], p[1]],
          [p[2], p[3]]
      ], sizing_mode='stretch_both')
  elif (len(p) == 5):
      l = grid([
          [p[0]],
          [p[1]],
          [p[2]],
          [p[3], p[4]],
      ], sizing_mode='stretch_both')
  # l.sizing_mode = 'stretch_both'
  return l

def addPlotXY(p, x, y, colour, plotType='line', legendLabel='None', yaxisSelection='default'):
  y.name = 'y'

  df = pd.concat([x, y], axis=1)
  hoverY = '@' + str(df.columns[1])
  hoverX = '@' + df.columns[0]
  source = ColumnDataSource(df)
  
  if plotType == 'line':
    p.line(df.columns[0], df.columns[1], legend_label=legendLabel, color=colour, line_width=2, source=source, y_range_name=yaxisSelection)
  elif plotType == 'circle':
    p.circle(df.columns[0], df.columns[1], legend_label=legendLabel, color=colour, size=5, alpha=0.5, source=source, y_range_name=yaxisSelection)
  
  addHoverInfo(p, hoverY, hoverX)
  p.legend.visible = False
  p.axis.major_label_text_font_size = '12pt'
  p.axis.axis_label_text_font_size = '12pt'
  p.title.text_font_size = '12pt'

  return p

def plotXY(x, y, title, xLabel, yLabel, colour, plotType='line', legendLabel='None'):
  p = figure(title=title, x_axis_label=xLabel, y_axis_label=yLabel)
  return addPlotXY(p, x, y, colour, plotType=plotType, legendLabel=legendLabel)
  
def addLegend(p, legendLocation=None):
  p.legend.visible=True
  p.legend.label_text_font_size = '12pt'
  p.legend.click_policy="hide"
  if legendLocation != None:
      p.legend.location = legendLocation
  else:
      p.legend.location = "top_left"
  return p

def plotError(error):
  df = pd.DataFrame(error)
  df.columns = ['w1', 'w2', 'w3', 'v1', 'v2', 'v3']
  df.reset_index(inplace=True)
  title = 'Torque'
  xlabel = 'Relative time in seconds'
  ylabel = 'Torque in mNm'

  legend1 = 'Demanded'
  colour = 'red'
  y = df['w1']
  x = df['index'].to_frame(name='index')
  p = plotXY(x, y, title, xlabel, ylabel, colour, legendLabel=legend1)

  # if len(dfMotives) > 0:
  #   torqueMeasure = dfMotives["positiveAxleLeftWheelTorqueInMilliNewtonMetres"]
  #   legend2 = 'Measured'
  #   colour = 'blue'
  #   x = getXAxisTime(dfMotives, start, timeFormat)
  #   y = torqueMeasure*-1
  #   addPlotXY(p, x, y, colour, legendLabel=legend2)

  p = addLegend(p, 'top_right')
  p.legend.visible=True
  return p 

if __name__ == "__main__":
  paramsFile = 'params.yaml'
  outputFile = 'output.csv'
  errorFile = 'error.csv'

  outputConfigs = list()
  errorTwists = list()

  with open(paramsFile) as file:
    botParams = yaml.load(file, Loader=yaml.FullLoader)
  
  bot = Bot(botParams)

  trajs = bot.generateReferenceTrajectory()

  # Tse_d   - Desired configuration
  # Tse_dn  - Desired next configuration
  time = 0.0
  bot.updateConfigOutputList(outputConfigs)

  # traj is now a list of numpy arrays.
  for traj, gripperState in trajs:
    for conf_no in range(len(traj)-1):
      Tse_d   = traj[conf_no]
      Tse_dn = traj[conf_no + 1]
      
      errorTwist, u, thetaDot = bot.feedbackControl(Tse_d, Tse_dn)

      controls = {'arm' : list(thetaDot), 'wheel' : list(u)}
      new_config = bot.nextState(controls)
      new_config["gripper"] = gripperState
      bot.updateConfig(new_config)
      bot.updateConfigOutputList(outputConfigs)
      errorTwists.append(errorTwist)
      time = time + bot.dt

  print("Generating output config file\n")
  with open(outputFile, 'w', newline='') as csv_file:
    writer = csv.writer(csv_file, delimiter=',')
    for i in outputConfigs:
      writer.writerow(i)

  with open(errorFile, 'w', newline='') as csv_file:
    writer = csv.writer(csv_file, delimiter=',')
    for i in errorTwists:
      writer.writerow(np.round(i,4))

  p1 = plotError(errorTwists)
  l = bokPlot(p1)
  show(l)