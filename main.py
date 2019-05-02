import numpy as np
import pandas as pd
import measurements
import pickle
import argparse
from extractAllFeatures import *
import os

from mpi4py import MPI


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
n = comm.Get_size()


if rank ==0 :
  #argparse stuff
  parser = argparse.ArgumentParser(description='Extract light curve features from csv file.')

  parser.add_argument('--lcFile', help='Csv file with the light curve info.',required=True)
  parser.add_argument('--simsFile', help='Pickle file with simulated light curve ids.',required=True)
  parser.add_argument('--realsFile', help='Csv file with real light curve ids.',required=True)


  args = parser.parse_args()

  print('light curve info file: {}'.format(args.lcFile))
  print('sims file: {}'.format(args.simsFile))
  print('reals file: {}'.format(args.realsFile))

  #open files
  data = pd.read_pickle(args.lcFile)
  sims = pickle.load(open(args.simsFile,'rb'))
  reals = np.loadtxt(args.realsFile)

  #create directory to store features
  os.system('mkdir calculatedFeatures')

  #read files content
  simsIds = []

  for i in sims[0]:
      simsIds.append(i[0])

  for i in range(1,len(sims)):
      simsIds.append(sims[i][0])

  for r in reals:
      simsIds.append(r)

  simsIds = np.array(simsIds)

filters = ['sdssr', 'sdssg', 'sdssz', 'sdssi']
for singleFilter in filters:
    if rank ==0:
      print('processing filter {}'.format(singleFilter))
      singleFDf= data[data['bandpass']==singleFilter]

    

      isSN = singleFDf.isin(simsIds).ID.values

      singleFDf['SN'] = isSN

      print('raw:')
      print('there are a total of {} supernovas'.format(len(singleFDf[singleFDf.SN==True].ID.unique())))
      print('there are a total of {} non-supernovas'.format(len(singleFDf[singleFDf.SN==False].ID.unique())))


      SN = singleFDf[singleFDf.SN==True]
      NSN = singleFDf[singleFDf.SN==False]
      nSupernovas = len(SN.ID.unique())

      selectedNSN = NSN.ID.unique()

      NSN = NSN[NSN.ID.isin(selectedNSN)]

      allDf = pd.concat([NSN,SN])

      #reformat to fit diego pipeline
      allDf.index.name = 'observation_id'
      allDf = allDf.set_index(['ID'], append=True)
      allDf = allDf.reorder_levels(['ID', 'observation_id'])

      print('balancing NSN:')
      print('there are a total of {} supernovas'.format(len(allDf[allDf.SN==True].index.get_level_values('ID').unique())))
      print('there are a total of {} non-supernovas'.format(len(allDf[allDf.SN==False].index.get_level_values('ID').unique())))

      allDf.columns = renameCols(list(allDf.columns))
      print_num_ids_shape(allDf)

      # unique ids
      ids = allDf.index.get_level_values('ID').unique()

      #filter obs with at least 5 data points
      allDf = filter_light_curves(allDf,5)

      print('filtering >5:')
      print('there are a total of {} supernovas'.format(len(allDf[allDf.SN==True].index.get_level_values('ID').unique())))
      print('there are a total of {} non-supernovas'.format(len(allDf[allDf.SN==False].index.get_level_values('ID').unique())))

      #we dont need to oversample but need it to 
      #fit the pipeline later on
      #thats why we oversample with 0

      allDf = oversample(allDf, 0)
      print_num_ids_shape(allDf)

      print('after oversampling:')
      print('there are a total of {} supernovas'.format(len(allDf[allDf.SN==True].index.get_level_values('ID').unique())))
      print('there are a total of {} non-supernovas'.format(len(allDf[allDf.SN==False].index.get_level_values('ID').unique())))


      SN = allDf[allDf.SN==True]
      NSN = allDf[allDf.SN==False]

      numSN  = len(SN.index.get_level_values('ID').unique())
      nsnIds = NSN.index.get_level_values('ID').unique().values

      #for now, get twice the amount of NSN than SN
      chosenNSNIds = np.random.choice(nsnIds,numSN*2,replace=False)

      NSN = NSN[NSN.index.get_level_values('ID').isin(chosenNSNIds)]

      allDf = allDf = pd.concat([NSN,SN])

      print('after reducing NSN:')
      print('there are a total of {} supernovas'.format(len(allDf[allDf.SN==True].index.get_level_values('ID').unique())))
      print('there are a total of {} non-supernovas'.format(len(allDf[allDf.SN==False].index.get_level_values('ID').unique())))

      # --------------------------------------------
      uniqueIds = allDf.index.get_level_values('ID').unique()
      numIds = len(uniqueIds)
      idChunks = [uniqueIds[i*numIds//n:(i+1)*numIds//n] for i in range(0, n)]
      assert len(uniqueIds) == sum([len(x) for x in idChunks])
      

      # --------------------------------------------
    else:
      allDf = None
      idChunks = None

    print('rank {} processing'.format(rank))

    allDf = comm.bcast(allDf, root=0)
    idChunks = comm.bcast(idChunks, root=0)

    myChunk = allDf[allDf.index.get_level_values('ID').isin(idChunks[rank])]
    #extract features
    myFeats = extract_features(myChunk)

    print('rank {} processing done'.format(rank))

    all_results = comm.gather(myFeats, root=0)

    #save features
    if rank==0:
      all_feats = pd.concat(all_results)
      all_feats.to_csv('./calculatedFeatures/features_{}.csv'.format(singleFilter))

