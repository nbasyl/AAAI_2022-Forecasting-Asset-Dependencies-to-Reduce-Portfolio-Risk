#### group0 There are in total 10 groups of dataset, please refer to the paper for detail.
## conv3d
/usr/bin/python3 /tmp2/syliu/adnn/src/main_moe_random.py --method conv3d_1 --save_model True --project_name us_baseline_g0 --dir_folder us_conv3d_g0 --dataset us --random_select True --gpu_id 0 --group_id 0

## lstm
/usr/bin/python3 /tmp2/syliu/adnn/src/main_moe_random.py --method lstm --save_model True --project_name us_baseline_g0 --dir_folder us_lstm_g0 --dataset us --random_select True --gpu_id 0 --group_id 0

## raw convlstm
/usr/bin/python3 /tmp2/syliu/adnn/src/main_moe_random.py --method Raw-ConvLSTM --save_model True --project_name us_baseline_g0 --dir_folder us_raw_convlstm_g0 --dataset us --random_select True --gpu_id 0 --group_id 0

## Clustering
/usr/bin/python3 /tmp2/syliu/adnn/src/main_moe_random.py --method cluster --save_model True --project_name us_baseline_g0 --dir_folder us_cluster_g0 --dataset us --random_select True --gpu_id 0 --group_id 0

## MLP CONVLSTM
/usr/bin/python3 /tmp2/syliu/adnn/src/main_moe_random.py --method mlp_1 --save_model True --project_name us_baseline_g0 --dir_folder us_mlp_g0 --dataset us --random_select True --gpu_id 0 --group_id 0

## T-ConvLSTM
/usr/bin/python3 /tmp2/syliu/adnn/src/main_moe_random.py --method tran --save_model True --project_name us_baseline_g0 --dir_folder us_TRAN_g0 --dataset us --random_select True --gpu_id 0 --group_id 0

## ADNN
/usr/bin/python3 /tmp2/syliu/adnn/src/main_moe_random.py --method moe_1 --save_model True --project_name us_adnn_e8t8_g0 --dir_folder us_adnn_e8t8_g0 --n_e 8 --top_k 8 --dataset us --random_select True --gpu_id 0 --group_id 0
