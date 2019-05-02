echo "Running extraction pipeline..."
python main.py --lcFile ../all_results_simulated.pkl --simsFile ../simulated_matches.pickle --realsFile ../realSNS.txt
mpirun -n 4 python calculateRespos.py --lcFile ../all_results_simulated.pkl 
echo "Finished."