import astropy.coordinates as coord
import astropy.units as u
import json

import pickle
import numpy as np
import re


from mpi4py import MPI


#output_dir="light_curves_candidates"
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
n = comm.Get_size()

if rank==0:
  inputs = range(0,10000)
else: 
  inputs = None

#ENVOI / RECEPTION => on ne s'embete pas avec un scatter (choix volontaire)...
lien_a_repartir = comm.bcast(inputs, root=0)

#CHAQUE PROCESS TRAITE SES FICHIERS SLAVE&MASTER
maxiKeyIndex = len(lien_a_repartir)

res = []
for ii in range(rank, maxiKeyIndex, n):
	res.append(lien_a_repartir[ii]+1)
		
#print "I am " + str(rank) +" andmy file size is " + str(len(complete_table))a

all_results = comm.gather(res, root=0)

if rank==0:
	print(res)