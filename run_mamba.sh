####################################################### demo ####################################################### 

# generate models for loop
python3 main.py -config=demo_lp -model=mamba --save_dir=./logs/demo --suffix=demo_lp_mamba_l1 -l=1
python3 main.py -config=demo_lp -model=mamba --save_dir=./logs/demo --suffix=demo_lp_mamba_l3 -l=3
python3 main.py -config=demo_lp -model=tf --save_dir=./logs/demo --suffix=demo_lp_tf_lin_l1 -l=1
python3 main.py -config=demo_lp -model=tf --save_dir=./logs/demo --suffix=demo_lp_tf_lin_l3 -l=3
python3 main.py -config=demo_lp -model=tf --save_dir=./logs/demo --suffix=demo_lp_tf_act_l1 -l=1 --activation=softmax
python3 main.py -config=demo_lp -model=tf --save_dir=./logs/demo --suffix=demo_lp_tf_act_l3 -l=3 --activation=softmax
python3 main.py -config=demo_lp -model=s4 --save_dir=./logs/demo --suffix=demo_lp_s4_l1 -l=1
python3 main.py -config=demo_lp -model=s4 --save_dir=./logs/demo --suffix=demo_lp_s4_l3 -l=3

# generate models for boyan chain
python3 main.py -config=demo_bc -model=mamba --save_dir=./logs/demo --suffix=demo_bc_mamba_l1 -l=1
python3 main.py -config=demo_bc -model=mamba --save_dir=./logs/demo --suffix=demo_bc_mamba_l3 -l=3
python3 main.py -config=demo_bc -model=tf --save_dir=./logs/demo --suffix=demo_bc_tf_lin_l1 -l=1
python3 main.py -config=demo_bc -model=tf --save_dir=./logs/demo --suffix=demo_bc_tf_lin_l3 -l=3
python3 main.py -config=demo_bc -model=tf --save_dir=./logs/demo --suffix=demo_bc_tf_act_l1 -l=1 --activation=softmax
python3 main.py -config=demo_bc -model=tf --save_dir=./logs/demo --suffix=demo_bc_tf_act_l3 -l=3 --activation=softmax
python3 main.py -config=demo_bc -model=s4 --save_dir=./logs/demo --suffix=demo_bc_s4_l1 -l=1
python3 main.py -config=demo_bc -model=s4 --save_dir=./logs/demo --suffix=demo_bc_s4_l3 -l=3

# run demo script
python3 demo.py -config=loop -l=1 -model mamba tf_lin tf s4 -path logs/demo/demo_lp_mamba_l1/seed_0/model.pth logs/demo/demo_lp_tf_lin_l1/seed_0/state_dict.pth logs/demo/demo_lp_tf_act_l1/seed_0/state_dict.pth logs/demo/demo_lp_s4_l1/seed_0/state_dict.pth --filename=demo_lp_l1
python3 demo.py -config=loop -l=3 -model mamba tf_lin tf s4 -path logs/demo/demo_lp_mamba_l3/seed_0/model.pth logs/demo/demo_lp_tf_lin_l3/seed_0/state_dict.pth logs/demo/demo_lp_tf_act_l3/seed_0/state_dict.pth logs/demo/demo_lp_s4_l3/seed_0/state_dict.pth --filename=demo_lp_l3
python3 demo.py -config=boyan -l=1 -model mamba tf_lin tf s4 -path logs/demo/demo_bc_mamba_l1/seed_0/model.pth logs/demo/demo_bc_tf_lin_l1/seed_0/state_dict.pth logs/demo/demo_bc_tf_act_l1/seed_0/state_dict.pth logs/demo/demo_bc_s4_l1/seed_0/state_dict.pth --filename=demo_bc_l1
python3 demo.py -config=boyan -l=3 -model mamba tf_lin tf s4 -path logs/demo/demo_bc_mamba_l3/seed_0/model.pth logs/demo/demo_bc_tf_lin_l3/seed_0/state_dict.pth logs/demo/demo_bc_tf_act_l3/seed_0/state_dict.pth logs/demo/demo_bc_s4_l3/seed_0/state_dict.pth --filename=demo_bc_l3

####################################################### boyan chain ####################################################### 

# mamba experiments
python3 main.py -config=boyan --representable -model=mamba -l=1 --suffix=bc_mamba_l1
python3 main.py -config=boyan --representable -model=mamba -l=3 --mode=auto --suffix=bc_mamba_l3
python3 main.py -config=boyan --representable -model=mamba -l=3 --mode=sequential --suffix=bc_mamba_l3_seq

# linear tf experiments
python3 main.py -config=boyan --representable -model=tf -l=1 --suffix=bc_tf_lin_l1
python3 main.py -config=boyan --representable -model=tf -l=3 --mode=auto --suffix=bc_tf_lin_l3
python3 main.py -config=boyan --representable -model=tf -l=3 --mode=sequential --suffix=bc_tf_lin_l3_seq

# nonlinear tf experiments
python3 main.py -config=boyan --representable -model=tf -l=1 --activation=softmax --suffix=bc_tf_act_l1
python3 main.py -config=boyan --representable -model=tf -l=3 --activation=softmax --mode=auto --suffix=bc_tf_act_l3
python3 main.py -config=boyan --representable -model=tf -l=3 --activation=softmax --mode=sequential --suffix=bc_tf_act_l3_seq

# s4 experiments
python3 main.py -config=boyan --representable -model=s4 -l=1 --suffix=bc_s4_l1
python3 main.py -config=boyan --representable -model=s4 -l=3 --mode=auto --suffix=bc_s4_l3
python3 main.py -config=boyan --representable -model=s4 -l=3 --mode=sequential --suffix=bc_s4_l3_seq

####################################################### cart pole ####################################################### 

# mamba experiments
python3 main.py -config=cartpole --representable -model=mamba -l=1 --suffix=cp_mamba_l1
python3 main.py -config=cartpole --representable -model=mamba -l=3 --mode=auto --suffix=cp_mamba_l3
python3 main.py -config=cartpole --representable -model=mamba -l=3 --mode=sequential --suffix=cp_mamba_l3_seq

# linear tf experiments
python3 main.py -config=cartpole --representable -model=tf -l=1 --suffix=cp_tf_lin_l1
python3 main.py -config=cartpole --representable -model=tf -l=3 --mode=auto --suffix=cp_tf_lin_l3
python3 main.py -config=cartpole --representable -model=tf -l=3 --mode=sequential --suffix=cp_tf_lin_l3_seq

# nonlinear tf experiments
python3 main.py -config=cartpole --representable -model=tf -l=1 --activation=softmax --suffix=cp_tf_act_l1
python3 main.py -config=cartpole --representable -model=tf -l=3 --activation=softmax --mode=auto --suffix=cp_tf_act_l3
python3 main.py -config=cartpole --representable -model=tf -l=3 --activation=softmax --mode=sequential --suffix=cp_tf_act_l3_seq

# s4 experiments
python3 main.py -config=cartpole --representable -model=s4 -l=1 --suffix=cp_s4_l1
python3 main.py -config=cartpole --representable -model=s4 -l=3 --mode=auto --suffix=cp_s4_l3
python3 main.py -config=cartpole --representable -model=s4 -l=3 --mode=sequential --suffix=cp_s4_l3_seq

####################################################### mountain car ####################################################### 

# mamba experiments
python3 main.py -config=mountaincar --representable -model=mamba -l=1 --suffix=mc_mamba_l1
python3 main.py -config=mountaincar --representable -model=mamba -l=3 --mode=auto --suffix=mc_mamba_l3
python3 main.py -config=mountaincar --representable -model=mamba -l=3 --mode=sequential --suffix=mc_mamba_l3_seq

# linear tf experiments
python3 main.py -config=mountaincar --representable -model=tf -l=1 --suffix=mc_tf_lin_l1
python3 main.py -config=mountaincar --representable -model=tf -l=3 --mode=auto --suffix=mc_tf_lin_l3
python3 main.py -config=mountaincar --representable -model=tf -l=3 --mode=sequential --suffix=mc_tf_lin_l3_seq

# nonlinear tf experiments
python3 main.py -config=mountaincar --representable -model=tf -l=1 --activation=softmax --suffix=mc_tf_act_l1
python3 main.py -config=mountaincar --representable -model=tf -l=3 --activation=softmax --mode=auto --suffix=mc_tf_act_l3
python3 main.py -config=mountaincar --representable -model=tf -l=3 --activation=softmax --mode=sequential --suffix=mc_tf_act_l3_seq

# s4 experiments
python3 main.py -config=mountaincar --representable -model=s4 -l=1 --suffix=mc_s4_l1
python3 main.py -config=mountaincar --representable -model=s4 -l=3 --mode=auto --suffix=mc_s4_l3
python3 main.py -config=mountaincar --representable -model=s4 -l=3 --mode=sequential --suffix=mc_s4_l3_seq
