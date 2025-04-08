####################################################### demo ####################################################### 

# generate models for loop
python3 main.py -config=demo -model=mamba --save_dir=./logs/demo --suffix=demo_mamba_l1 -l=1
python3 main.py -config=demo -model=mamba --save_dir=./logs/demo --suffix=demo_mamba_l3 -l=3
python3 main.py -config=demo -model=tf --save_dir=./logs/demo --suffix=demo_tf_l1 -l=1 --activation=softmax
python3 main.py -config=demo -model=tf --save_dir=./logs/demo --suffix=demo_tf_l3 -l=3 --activation=softmax
python3 main.py -config=demo -model=s4 --save_dir=./logs/demo --suffix=demo_s4_l1 -l=1
python3 main.py -config=demo -model=s4 --save_dir=./logs/demo --suffix=demo_s4_l3 -l=3

# run demo script
python3 demo.py -l=1 -model mamba tf s4 -path logs/demo/demo_mamba_l1/seed_0/model.pth logs/demo/demo_tf_l1/seed_0/state_dict.pth logs/demo/demo_s4_l1/seed_0/state_dict.pth --filename=demo_l1
python3 demo.py -l=3 -model mamba tf s4 -path logs/demo/demo_mamba_l3/seed_0/model.pth logs/demo/demo_tf_l3/seed_0/state_dict.pth logs/demo/demo_s4_l3/seed_0/state_dict.pth --filename=demo_l3

####################################################### boyan chain ####################################################### 

# mamba experiments
python3 main.py -config=boyan --representable -model=mamba -l=1 --suffix=bc_mamba_l1 ;
python3 main.py -config=boyan --representable -model=mamba -l=3 --mode=auto --suffix=bc_mamba_l3

# tf experiments
python3 main.py -config=boyan --representable -model=tf -l=1 --activation=softmax --suffix=bc_tf_l1 ;
python3 main.py -config=boyan --representable -model=tf -l=3 --activation=softmax --mode=auto --suffix=bc_tf_l3

# s4 experiments
python3 main.py -config=boyan --representable -model=s4 -l=1 --suffix=bc_s4_l1 ;
python3 main.py -config=boyan --representable -model=s4 -l=3 --mode=auto --suffix=bc_s4_l3

####################################################### cart pole ####################################################### 

# mamba experiments
python3 main.py -config=cartpole --representable -model=mamba -l=1 --suffix=cp_mamba_l1 ;
python3 main.py -config=cartpole --representable -model=mamba -l=3 --mode=auto --suffix=cp_mamba_l3

# tf experiments
python3 main.py -config=cartpole --representable -model=tf -l=1 --activation=softmax --suffix=cp_tf_l1 ;
python3 main.py -config=cartpole --representable -model=tf -l=3 --activation=softmax --mode=auto --suffix=cp_tf_l3


####################################################### mountain car ####################################################### 

# mamba experiments
python3 main.py -config=mountaincar --representable -model=mamba -l=1 --suffix=mc_mamba_l1 ;
python3 main.py -config=mountaincar --representable -model=mamba -l=3 --mode=auto --suffix=mc_mamba_l3

# tf experiments
python3 main.py -config=mountaincar --representable -model=tf -l=1 --activation=softmax --suffix=mc_tf_l1 ;
python3 main.py -config=mountaincar --representable -model=tf -l=3 --activation=softmax --mode=auto --suffix=mc_tf_l3
