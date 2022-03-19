# ADNN: Forecasting Asset Dependencies to Reduce Portfolio Risk

This is our implementation of the paper:

*Haoren Zhu <sup>†</sup>, Shih-Yang Liu <sup>†</sup>, Pengfei Zhao, Yingying Chen, Dik Lun Lee. 2022. [Forecasting Asset Dependencies to Reduce Portfolio Risk](https://www.aaai.org/AAAI22Papers/AAAI-7053.ZhuH.pdf) In AAAI'22.*

**Please cite our AAAI'22 paper if you use our codes. Thanks!**
```
@article{zhu2022forecasting,
  title={Forecasting Asset Dependencies to Reduce Portfolio Risk},
  author={Zhu, Haoren and Liu, Shih-Yang and Zhao, Pengfei and Chen, Yingying and Lee, Dik Lun},
  year={2022}
}
```

## Example to run the codes		

Train and evaluate our model:

```
python3 /tmp2/syliu/adnn/src/main_moe_random.py --method moe_1 --save_model True --project_name us_adnn_e8t8_g0 --dir_folder us_adnn_e8t8_g0 --n_e 8 --top_k 4 --dataset us --random_select True --gpu_id 0 --group_id 0
```

## Reproducibility

```
Please refer to the technical appendix for the hyper-parameters setting. In general, we found out that setting number of experts to 8 and top k selection to 4 yields the better performance
```

## Suggestions for parameters

Several important parameters need to be tuned for different applications, which are:

```
parser.add_argument('--num_asset',default=32,type=int) ## This is the total number of assets, you wish to construct the portfolio
parser.add_argument('--n_clusters',default=12,type=int) ## The number of clusters for the clustering reposition algorithm introduced in the paper, aka, P-Convlstm
parser.add_argument('--input_length',default=10,type=int) ## The length of the input ADM sequences
parser.add_argument('--input_gap',default=21,type=int) ## The time gap between each ADM of a ADM input sequences, the u in the below figure 
parser.add_argument('--horizon',default=21,type=int) ## The time between the last ADM of the input sequences and the target future ADM. For example, if the portfolio is rebalanced on a monthly basis, the investors are interested in the ADM one month later, and horizon should be set to 21
parser.add_argument("--lag_t", type = int, default=42)  ## The time you wish to calculate the ADM, the n_lag in the below figure

parser.add_argument('--dilated', type=int, default=0) ## check the definition of dilated convolution, 0 is not to use
parser.add_argument("--filter_size", type=int, default=5) ## the convolution kernel size

parser.add_argument('--n_e', type=int, default=8) ## the number of experts for MoE
parser.add_argument('--top_k', type=int, default=4) ## the top-k selection for MoE
```

<img src="https://i.imgur.com/P4QIxax.png" alt="Alt text" title="ADMs construction">

