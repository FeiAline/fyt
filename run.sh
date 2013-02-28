#!/bin/bash

# general instruction for program to run 
../Release/cudaGabor -d  ../Release/det_RP5_profile_left_18sz_mika_para.xml -m ../Release/det_ene_RP5_v_rmDC_fun_neu_o4_2.0_1.6_0.0_winnow.xml -f ../Release/ten.avi

# plot different graphs
./plot.gp
