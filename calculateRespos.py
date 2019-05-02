
import numpy as np
import pandas as pd
import pickle
import argparse
from mpi4py import MPI


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
n = comm.Get_size()

if rank == 0:

    print('rank 0 preprocessing')
    # argparse stuff
    parser = argparse.ArgumentParser(
        description='Extract light curve features from csv file.')
    parser.add_argument(
        '--lcFile', help='Csv file with the light curve info.', required=True)

    args = parser.parse_args()

    print('light curve info file: {}'.format(args.lcFile))

    # read data
    data = pd.read_pickle(args.lcFile)

    # get r filter
    rData = data[data['bandpass'] == 'sdssr']

    # set index
    rData.index.name = 'observation_id'
    rData = rData.set_index(['ID'], append=True)
    rData = rData.reorder_levels(['ID', 'observation_id'])

    uniqueIds = rData.index.get_level_values('ID').unique()
    numIds = len(uniqueIds)
    idChunks = [uniqueIds[i*numIds//n:(i+1)*numIds//n] for i in range(0, n)]
    assert len(uniqueIds) == sum([len(x) for x in idChunks])
    print('rank 0 preprocessing done')

else:
    rData = None
    idChunks = None

print('rank {} processing'.format(rank))

rData = comm.bcast(rData, root=0)
idChunks = comm.bcast(idChunks, root=0)


myChunk = rData[rData.index.get_level_values('ID').isin(idChunks[rank])]

# extract num of positive residues
numResPos = myChunk[myChunk.classification >= 6].groupby(
    myChunk[myChunk.classification >= 6].index.get_level_values('ID'))['classification'].count()

# get percentagge and save file
totObs = myChunk.groupby(myChunk.index.get_level_values('ID'))[
    'classification'].count()
respos = numResPos/totObs
respos = respos.fillna(0)

print('rank {} processing done'.format(rank))

all_results = comm.gather(respos, root=0)

if rank == 0:
    total = pd.concat(all_results)
    assert len(total) == len(uniqueIds)
    total.to_csv('./calculatedFeatures/respos_r.csv')
