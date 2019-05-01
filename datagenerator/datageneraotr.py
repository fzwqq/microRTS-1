import xml.etree.ElementTree as ET
import numpy as np

# dataframe
# element of trajector is a 5-dim tuple
# {"id", "time", "state", "action" , "reward", "state_next"} 
# id is state id
# time is the game timestamp of state

# state is not decided yet
# you could override encode_game_state in class Trace

# action is action list of unit decided in state
# action is a 4-dim tuple
# action is 4-dim tuple = { unit id , action type, direction, destination }
#   for attack action, destination is army x y 
#   for produce unit action destination x is meanless , y is the type of produced unit
#   for other action , based on unit x y and direction could know it destination

# reward function is not defined yet
# you could override get_reward in class Trace

# state next is represent as a state id
        



class UnitType():
    def __init__(self, unit_type_id, name, cost, hp, minDamage, maxDamage, attackRange, produceTime, moveTime, 
                attackTime, harvestTime, returnTime, harvestAmount, sightRadius, isResource, isStockpile, 
                canHarvest, canMove, canAttack):
        self.unit_type_id = unit_type_id
        self.name = name
        self.cost = cost
        self.hp = hp
        self.minDamage = minDamage
        self.maxDamage = maxDamage
        self.attackRange = attackRange
        self.produceTime = produceTime
        self.moveTime = moveTime
        self.attackTime = attackTime
        self.harvestTime = harvestTime
        self.returnTime = returnTime
        self.harvestAmount = harvestAmount
        self.sightRadius = sightRadius
        self.isResource = isResource
        self.isStockpile = isStockpile
        self.canHarvest = canHarvest
        self.canMove = canMove
        self.canAttack = canAttack


class Unit():
    def __init__(self, type_name, unit_id, player_id, x, y, resources, hp):
        self.type_name = type_name
        self.unit_id = unit_id
        self.player_id = player_id
        self.x = x
        self.y = y
        self.resources = resources
        self.hp = hp

class Player():
    def __init__(self, player_id, resources = 0, unit_list=None):
        self.player_id = player_id
        self.resources = resources
        self.unit_list = unit_list


class PhysicalGameState():
    def __init__(self, width, height, player1, player2, game_state):
        self.width = width
        self.height = height
        self.player1 = player1
        self.player2 = player2
        self.game_state = game_state



# Trace is a integrate game trajectory
class Trace():

    def __init__(self):
        self.unit_type_table = {}
        self.unit_dict = {}
        self.trajectory = []
        self.state_id_cnt = 0
        self.action_projection = { "0":"hold" , "1":"move", "2":"harvest", "3":"return", "4":"produce_unit", "5":"attack"}
        self.direction_projection = {"-1":"none", "0":"up", "1":"right", "2":"down","3":"left", "10":"none"}
    
    # not defined yet
    def get_reward(self, state, state_next):
        return 0

    # decode_unit_type_table
    def decode_unit_type_table(self,unit_type_table):
        for child in unit_type_table :
            attrib = child.attrib
            unit_type = UnitType(attrib["ID"], attrib["name"], attrib["cost"], attrib["hp"], attrib["minDamage"], attrib["maxDamage"], 
                    attrib["attackRange"], attrib['produceTime'], attrib['moveTime'], attrib['attackTime'], attrib["harvestTime"],
                    attrib["returnTime"], attrib["harvestAmount"], attrib["sightRadius"], attrib["isResource"], attrib["isStockpile"],
                    attrib["canHarvest"], attrib["canMove"], attrib["canAttack"])
            self.unit_type_table[unit_type.name] = unit_type
        

    # decode xml format pgs and actions to required format
    def append_trace_entry(self, timestamp, pgs, action):
        state = self.decode_pgs(pgs)
        state_next_id = -1
        if len(self.trajectory) == 0:
            state_next_id = -1
            reward = 0
        else:
            state_next = self.trajectory[-1]
            state_next_id = state_next["id"]
            reward = self.get_reward(state, state_next["state"])
    
        action_list = self.decode_actions(action)
        trace_entry = {"id" : self.state_id_cnt, "time" : timestamp,  "state" : state, "action":action_list, "reward":reward, "state_next":state_next_id} 
        self.state_id_cnt += 1
        self.trajectory.append(trace_entry)

    # decode xml fomat pgs
    def decode_pgs(self,pgs):
        self.unit_dict = {}
        width = int(pgs.attrib['width'])
        height = int(pgs.attrib['height'])
        terrain = pgs[0]

        players = pgs[1]
        player1 = Player(players[0].attrib["ID"], players[0].attrib["resources"])
        player2 = Player(players[1].attrib["ID"], players[1].attrib["resources"])

        units = pgs[2]
        unit_list_share = []
        unit_list_1 = []
        unit_list_2 = []
        for child in units:
            attrib = child.attrib
            unit = Unit(attrib["type"], attrib["ID"], attrib["player"], int(attrib["x"]), int(attrib["y"]), attrib["resources"], attrib["hitpoints"])
            self.unit_dict[unit.unit_id] = unit
        game_state = self.encode_game_state(width, height, terrain, self.unit_dict)
        return game_state

    def encode_game_state(self, width, height, terrain, unit_dict):
        # Initialization of spatial features
        spatial_features = np.zeros((18,height,width))
        #channel_wall
        channel_wall = spatial_features[0]
        for i in range(len(terrain)):
            row = i / width
            col = i % width
            if terrain[i] == 1:
                channel_wall[row][col] = 1

        channel_resource = spatial_features[1]
        channel_self_type = spatial_features[2:8]
        channel_self_hp = spatial_features[8]
        channel_self_resource_carried = spatial_features[9]
        channel_enemy_type = spatial_features[9:15]
        channel_enemy_hp = spatial_features[16]
        channel_enemy_resource_carried = spatial_features[17]
        for key in unit_dict:
            unit = unit_dict[key]
            _player = unit.player_id
            x = unit.x 
            y = unit.y
            unit_type_id = int(self.unit_type_table[unit.type_name].unit_type_id)
            # neutral
            if _player == "-1":
                channel_resource[x][y] = unit.resources
            elif _player == "0":
                # get the index of this type
                idx = unit_type_id
                channel_self_type[idx][x][y] = 1
                channel_self_hp[x][y] = unit.hp
                channel_self_resource_carried[x][y] = unit.resources
            else: 
                idx = unit_type_id
                channel_enemy_type[idx][x][y] = 1
                channel_enemy_hp[x][y] = unit.hp
                channel_enemy_resource_carried[x][y] = unit.resources
        #print(spatial_features)
        return spatial_features

    # decode xml actions
    # action is 4-dim tuple = { unit id , action type, direction, destination }
    # for attack action type destination is army x y 
    # for produce unit action destination x is meanless , y is the type of produced unit
    # for other action , based on unit x y and direction could know it destination
    def decode_actions(self, actions):
        action_list = []
        for action in actions:
            unit_id = action.attrib["unitID"]
            for unit_action in action:
                attrib = unit_action.attrib
                action_type = attrib["type"]
                if self.action_projection[action_type] == "attack":
                    x = attrib["x"]
                    y = attrib["y"]
                    direction = "-1"
                elif self.action_projection[action_type] == "produce_unit":
                    x = -1
                    y = self.unit_type_table[attrib["unitType"]].unit_type_id
                    direction = attrib["parameter"]
                else:
                    direction = attrib["parameter"]
                    x = -2
                    y = -2
                action_atom = {'id':unit_id, 'type':action_type, 'direction':direction, 'destination':[x,y]}
                action_list.append(action_atom)

        return action_list
    
    def get_trajectory(self):
        return self.trajectory
            



class Datagenerator():
    def __init__(self):
        self.trace_list = []

    # read xml to generate a trace instance
    # if mode = a means append
    # if mode = w means clear the traces and append new trace
    def get_trace_from_file(self,filename, mode):
        if mode == "w":
            self.traces = []
        trace = Trace()

        tree = ET.parse(filename)
        # root is rts.Trace
        root = tree.getroot()
        # rts_units_unitTypeTable is child of root
        # get unit type list
        # unit type define attributes of unit
        unit_type_table = root[0]
        trace.decode_unit_type_table(unit_type_table)

        entries = root[1]
        reversed_entries = []
        for child in entries:
            reversed_entries.append(child)
        reversed_entries.reverse()
        # reverse the entries so could generate state from end to begin
        # child of entries is TraceEntry
        for child in reversed_entries:
            timestamp = child.attrib["time"]
            pgs = child[0]
            actions = child[1]
            trace.append_trace_entry(timestamp, pgs, actions)
            trajectory = trace.get_trajectory()
        self.trace_list.append(trace)
    
    def write_traces_to_file(self,root_path):
        for trace in self.trace_list:
            trajectory = trace.get_trajectory()
            for entry in trajectory:
                with open(root_path,'a') as f:
                    f.write(str(entry))
                    f.write('\n\n\n')


if __name__ == "__main__":
    dg = Datagenerator()
    dg.get_trace_from_file("trace1.xml","a")
    dg.write_traces_to_file('test.data')


        


