#!/bin/bash


((Nfull = "$1"*"$2"))
((Nfullm1 = "$Nfull" - 1))
((Nper = "$1"))
((Nperm1 = "$1"-1))
((Njob = "$2"))
((Njobm1 = "$2"-1))

mkdir bash_out
#rm ../scan_summary/23032022_W01_AVEp0_VEp0_result_log.txt

for i in $(seq 0 "$Njobm1")
do
#    ((start = "$i"*"$Nper"))
#    ((end = "$i"*"$Nper" + "$Nper"))

  sbatch --array [0-"$Nperm1"] run_simulation.sh "$i"
done
