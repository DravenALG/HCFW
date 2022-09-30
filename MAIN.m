%% init the workspace
close all; clear; clc; warning off;

%% globel settings
train_param.current_bits = 32;
train_param.max_iter=5;

%% load dataset
train_param.ds_name='MIRFLICKR';  % NUSWIDE21  MIRFLICKR
train_param.load_type='full_label'; %full_label  new_label  all_new_label
[train_param,XTrain,LTrain,LTrain_All,XQuery,LQuery,LQuery_All,K,K_All] = load_dataset(train_param);

%% HCFW
train_param.current_hashmethod='HCFW';
OURparam = train_param;
OURparam.theta = 1;
OURparam.delta = 1;
[eva1,~] = evaluate_HCFW(XTrain,LTrain,XQuery,LQuery,K,OURparam);