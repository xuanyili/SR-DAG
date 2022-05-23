#!/usr/bin/env bash
# data_bn_list=(synthetic_gauss_n5_e7_0 synthetic_gauss_n5_e7_1 synthetic_gauss_n5_e7_2 synthetic_gauss_n16_e40_0 synthetic_gauss_n16_e40_1 synthetic_gauss_n16_e40_2)
# data_bn_list=(synthetic_bn_n40_e50_0)
data_list=(ct_bn_n60_e60_10000)
# data_list=(syn_n5_e10)
data_benchmark_list=(alarm insurance hepar2)
seed_list=(0)
model_list=(bnrl_bic bnrl_ctbic)
# model_list=(bnrl_bdeu gobnilp_bdeu gran_dag dag_gnn rlbic rlbic2 notear)
# model_list=(bnrl_bic bnrl_bdeu gobnilp_bdeu gobnilp_bic gran_dag dag_gnn notear pc rlbic rlbic2)
for model in "${model_list[@]}"
do
    for data in "${data_list[@]}"
    do
        python run.py --data ${data} --model ${model}
    done
done

# for data in "${data_list[@]}"
# do
#     for seed in ${seed_list[@]}
#     do
#         python run.py --data ${data}
#     done
# done

# for data in "${data_benchmark_list[@]}"
# do
#     for seed in ${seed_list[@]}
#     do
#         python run.py --data ${data}
#     done
# done