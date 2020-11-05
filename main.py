
# Utils

import sys
import os
import settings
from envs import env
import numpy as np
import time
from datetime import datetime
from pathlib import Path

# SQL
import sqlalchemy as db
import psycopg2
import json
import pickle
import zlib

import Database.Database as Database


### Buscamos experimentos pendientes
connect = Database.Database()


# Algorithms

from Metaheuristics.SCA_SCP import SineCosine_SCP
from Metaheuristics.WOAQL_SCP import WOAQL_SCP
from Metaheuristics.WOA_SCP import WOA_SCP

flag = True
while flag:

    id, algorithm, params = connect.getLastPendingAlgorithm('pendiente')
    
    if id == 0:
        print('No hay más ejecuciones pendientes')
        break
        
       
    print("------------------------------------------------------------------------------------------------------------------\n")
    print(f'Id Execution: {id} -  {algorithm}')
    print(json.dumps(params,indent=4))
    print("------------------------------------------------------------------------------------------------------------------\n")

    if(algorithm == 'WOA_SCP'):
        if  WOA_SCP(id,
                params['instance_file'],
                params['instance_dir'],
                params['population'],
                params['maxIter'],
                params['discretizationScheme'],
                params['repair']
                ) == True:
            print(f'Ejecución {id} completada ')

    if(algorithm == 'WOA_SCP_repair2'):
        if  WOA_SCP(id,
                params['instance_file'],
                params['instance_dir'],
                params['population'],
                params['maxIter'],
                params['discretizationScheme'],
                params['repair']
                ) == True:
            print(f'Ejecución {id} completada ')

    if(algorithm == 'WOA_SCP_repair3'):
        if  WOA_SCP(id,
                params['instance_file'],
                params['instance_dir'],
                params['population'],
                params['maxIter'],
                params['discretizationScheme'],
                params['repair']
                ) == True:
            print(f'Ejecución {id} completada ')

    if(algorithm == 'WOAQL_SCP'):
        if  WOAQL_SCP(id,
                params['instance_file'],
                params['instance_dir'],
                params['population'],
                params['maxIter'],
                params['discretizationScheme'],
                params['ql_alpha'],
                params['ql_gamma'],
                params['repair'],
                params['policy'],
                params['rewardType'],
                params['qlAlphaType']
                ) == True:
            print(f'Ejecución {id} completada ')

    if(algorithm == 'SCA_SCP'):
        if  SineCosine_SCP(id,
                params['instance_file'],
                params['instance_dir'],
                params['population'],
                params['maxIter'],
                params['discretizationScheme'],
                params['repair']
                ) == True:
            print(f'Ejecución {id} completada ')
  