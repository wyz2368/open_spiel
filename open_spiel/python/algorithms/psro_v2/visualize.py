"""
For simultaneous move game(laser tag, markov soccer), perform rollout for strategies required, record the trajectories. Play it in pygame if asked. Assume the directory structure is the standard directory structure
"""
import os
import os.path as op
from absl import app
from absl import flags
try:
  import pyspiel
  from open_spiel.python.algorithms.psro_v2.combined_game import load_env
  from open_spiel.python.algorithms.psro_v2.eval_utils import load_strategy,load_pkl
  from open_spiel.python.algorithms.psro_v2.utils import sample_strategy
except:
  print("pyspiel not installed on local machine")
from math import floor
import pygame
pygame.font.init()
import functools
print = functools.partial(print, flush=True)

FLAGS = flags.FLAGS

# rollout related flags
flags.DEFINE_bool("rollout_record", True, "rollout game and record")
flags.DEFINE_list("checkpoint_dir_list","","root directories that contains strategies ")
flags.DEFINE_list("number_iters_at_each_dir","","the nash equilibrium of which iteration to rollout for each directory")
flags.DEFINE_bool("rollout_nash",True,"rollout nash strategies or not")

# visualization related flags
flags.DEFINE_bool("visualize", False, "Game name.")
flags.DEFINE_list("list_of_rollout_txt","","the list of txt file to visualize")
flags.DEFINE_bool("markov_soccer",True,"visualize markov soccer or laser tag")
flags.DEFINE_integer("width",1280,"width of the pygame display panel")
flags.DEFINE_integer("height",720,"width of the pygame display panel")

ZERO_THRESHOLD = 0.001

def load_env_strategies(strat_root_dir, it, rollout_nash, ne_dir=None):
  """
  strat_root_dir : checkpoint_dir/strategies/
  it: gpsro iteration
  rollout_nash: load nash strategies at the iteration or pure strategy at it
  ne_dir: root directory for nash equilibrium
  """
  player_dir = [op.join(strat_root_dir,x) \
      for x in os.listdir(strat_root_dir) \
      if op.isdir(os.path.join(strat_root_dir,x))]
  
  # initialize game & rl environment & strategy type
  env = load_env(strat_root_dir, "env.pkl") 
  strategy_kwargs = load_pkl(op.join(strat_root_dir, "kwargs.pkl"))
  strategy_type = strategy_kwargs.pop("policy_class")
  
  strats = []
  for p in range(len(player_dir)):
    strategy_player = []
    if not rollout_nash: # load the strategies of iteration and return
      weight = load_pkl(player_dir[p]+"/"+str(it)+".pkl")
      strategy = load_strategy(strategy_type,
                               strategy_kwargs,
                               env, p, weight)
      strategy_player.append(strategy)
      strats_prob = [[1],[1]]
    else: # read out nash strategies and return
      if p == 0: #load nash
        nash_ne = load_pkl(op.join(ne_dir,str(it)+".pkl"))
        strats_prob = [[x for x in ele \
            if x>ZERO_THRESHOLD] for ele in nash_ne]

      for weight_file in range(1,it+2): # 1st it two weight file
        if nash_ne[p][weight_file-1] < ZERO_THRESHOLD:
          continue # only combine nash strategies
        weight = load_pkl(player_dir[p]+"/"+str(weight_file)+".pkl")
        strategy = load_strategy(strategy_type,
                                 strategy_kwargs,
                                 env, p, weight)
        strategy_player.append(strategy)
    strats.append(strategy_player)

  return env, strats, strats_prob

GREEN = (0,255,0)
RED = (255,0,0)
DARK_RED = (255,165,165)
WHITE = (255,255,255)
BISQUE = (255,228,196)
TURQUOISE = (0,197,205)
BLACK = (0,0,0)
BLUE = (0,0,255)
GREY=(211,211,211)

def init_background_surface(row_num, col_num, display, markov_soccer=True):
  """
  Return:
    surface: the smaller surface to work upon
    offset : offset of the surface with regards to display
    txt_width: in the smaller surface, how much width the txt uses up
    txt_height: in the smaller surface, how much height the txt uses up
    row_height: height of each row
    column_widht: width of each column
  """
  # surface takes up a proportion of the whole display for asthetical reasons
  w,h = display.get_size()
  offset = (floor(w/8),floor(h/8))
  w, h = w-2*offset[0], h-2*offset[1]
  surface = pygame.Surface((w,h))
  surface.fill(BLACK)
  # txt regions on the top of the surface
  txt_width, txt_height = w,floor(h/6)
  pygame.draw.rect(surface, BLACK, pygame.Rect(0,0,txt_width,txt_height))
  row_height = floor((h-txt_height)/row_num)
  column_width = floor(w/col_num)
  # draw grid
  for i in range(row_num+1):
    pygame.draw.line(surface,WHITE,(0,txt_height+i*row_height),
                                   (w,txt_height+i*row_height),5)
  for i in range(col_num+1):
    pygame.draw.line(surface,WHITE,(i*column_width,txt_height),
                                   (i*column_width,h),5)
    
  return surface, offset, [txt_width,txt_height], row_height, column_width


def visualize(fi_path, display, markov_soccer=True):
  """
  """
  rewards_font = pygame.font.SysFont('Comic Sans MS', 32)
  character_font = pygame.font.SysFont('Comic Sans MS', 64)
  with open(fi_path,'r') as fi:
    whole_file = fi.read()
  episodes = whole_file.split("$$")[:-1] # last split is \n
  sample_grid = episodes[1].split("$\n")[0].split('\n')
  sample_grid = [x for x in sample_grid if x != '']
  row_num = len(sample_grid)
  col_num = len(sample_grid[0])
  grid, offset, txt_dim, rh, cw = init_background_surface(row_num, col_num, display, markov_soccer)
  rewards = [0,0]
  num_episode = 0
  for episode in episodes:
    num_episode += 1
    episode = episode[1:] if episode[0]=='\n' else episode # remove \n at the start of episodes
    episode_str = episode.split("$\n")
    reward = episode_str[-1].split("_")
    reward = [float(reward[1]),float(reward[2])]
    rewards = [rewards[0]+reward[0], rewards[1]+reward[1]]
    info_str_cur = "Cur Rewards: %5.2f %5.2f"%(reward[0],reward[1])
    info_str_avg = "Avg Rewards: %5.2f %5.2f"%(rewards[0]/num_episode,rewards[1]/num_episode)
    
    epi_surface = rewards_font.render("episode "+str(num_episode),True,GREY)
    info_surface = rewards_font.render(info_str_cur,True,GREY)
    info_avg_surface = rewards_font.render(info_str_avg,True,GREY)
    
    episode_str = episode_str[:-1]    # remove reward information for episode
    for time_step in range(len(episode_str)):
      chars = ['a','b','A','B','O']
      display.blit(grid,offset)
      display.blit(info_surface,(offset[0]+10,offset[1]+10))
      display.blit(info_avg_surface,(offset[0]+10,offset[1]+floor(txt_dim[1]/2)))
      display.blit(epi_surface,(offset[0]+400,offset[1]+10))
      for ele in chars:
        index = episode_str[time_step].find(ele)
        if index < 0:
          continue
        pos = (index%(col_num+1),index//(col_num+1)) #+1 because of the \n at the end of each line
        center = (offset[0]+floor((pos[0]+0.5)*cw),offset[1]+txt_dim[1]+floor((pos[1]+0.5)*rh))
        if ele in ['A','B','O']:
          pygame.draw.circle(display, GREY, center, floor(min(rh,cw)/2-5), 8)
        if ele == 'A':
          character = character_font.render('a',True,GREY)
        elif ele == 'B':
          character = character_font.render('b',True,GREY)
        elif ele == 'O':
          continue
        else:
          character = character_font.render(ele,True,GREY)  
        display.blit(character,(center[0]-floor(0.12*cw),center[1]-floor(0.12*rh)))
      pygame.display.flip()
      pygame.time.wait(500)
      if time_step == 0:
        pygame.time.wait(200)
      display.fill(GREY)
    
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        pygame.display.quit()
        return
          

def rollout_and_record(di, it, rollout_nash=True, sims_per_entry=100):
  """
  Params:
    it: number of iteration
    di: checkpoint_directory
    rollout_nash: rolling out nash equilibrium policy by iteration it, or just rollout the policy stated at that iteration
    
  """
  env, strats, strats_prob = load_env_strategies(di+"/strategies",
                                                 it,rollout_nash,
                                                 di+"/nash_prob")
  fi_path = os.path.join(di,str(it)+'_rollout_nash_'+str(rollout_nash)+'.txt')
  fi = open(fi_path,'w')
  for _ in range(sims_per_entry):
    agents = sample_strategy(strats, strats_prob, probs_are_marginal=True)
    this_episode_str = ""
    time_step = env.reset()
    assert env.get_state.observation_string,"game has to support observation"
    while not time_step.last():
      this_episode_str += env.get_state.observation_string(0)+"$\n"
      if time_step.is_simultaneous_move():
        action_list = []
        for agent in agents:
          output = agent.step(time_step, is_evaluation=True)
          action_list.append(output.action)
      else:
        player_id = time_step.observations["current_player"]
        agent_output = agents[player_id].step(time_step, is_evaluation=True)
        action_list = [agent_output.action]
      time_step = env.step(action_list)
    rewards = env.get_state.returns()
    this_episode_str += "\nRewards_"
    for ele in rewards:
      this_episode_str += str(ele)+"_"
    this_episode_str = this_episode_str[:-1]+"$$\n"
    fi.write(this_episode_str)

  fi.close()
  print("rollout dumbed to",fi_path)
  return fi_path

def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  if FLAGS.rollout_record:
    assert len(FLAGS.checkpoint_dir_list) == len(FLAGS.number_iters_at_each_dir)  
    number_iters_at_each_dir = [int(x) for x in FLAGS.number_iters_at_each_dir]
    for directory,it in zip(FLAGS.checkpoint_dir_list,number_iters_at_each_dir):
      rollout_and_record(directory, it, FLAGS.rollout_nash)

  if FLAGS.visualize:
    pygame.init()
    pygame.font.init()
    display =  pygame.display.set_mode((FLAGS.width,FLAGS.height),pygame.HWSURFACE|pygame.DOUBLEBUF)
    display.fill(GREY)
    import pdb
    pdb.set_trace()
    for fi in FLAGS.list_of_rollout_txt:
      visualize(fi, display, markov_soccer=FLAGS.markov_soccer)
    
    pygame.quit()

if __name__ == "__main__":
  app.run(main)


