# 查看当前工作目录
import os
retval = os.getcwd()
print("当前工作目录为: %s" % retval)

# 添加环境变量--myyyPLM的路径
import sys 
sys.path.extend(["/home/ZOE_BERT/V1-2_InputAdd/OpenKE-PyTorch","/home/ZOE_BERT"])
print(sys.path)

import openke
from openke.config import Trainer, Tester
from openke.module.model import TransH
from openke.module.loss import MarginLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader

# dataloader for training
train_dataloader = TrainDataLoader(
	in_path = "/home/ZOE_BERT/V1-2_InputAdd/13_35_43/", 
	nbatches = 100,
	threads = 8, 
	sampling_mode = "normal", 
	bern_flag = 1, 
	filter_flag = 1, 
	neg_ent = 25,
	neg_rel = 0)

# dataloader for test
test_dataloader = TestDataLoader("/home/ZOE_BERT/V1-2_InputAdd/13_35_43/", "link")

# define the model
transh = TransH(
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
	dim = 200, 
	p_norm = 1, 
	norm_flag = True)

# define the loss function
model = NegativeSampling(
	model = transh, 
	loss = MarginLoss(margin = 4.0),
	batch_size = train_dataloader.get_batch_size()
)


# train the model
trainer = Trainer(model = model, data_loader = train_dataloader, train_times = 1000, alpha = 0.5, use_gpu = True)
trainer.run()
transh.save_checkpoint('./checkpoint/1k/transh.ckpt')

# test the model
transh.load_checkpoint('./checkpoint/1k/transh.ckpt')
tester = Tester(model = transh, data_loader = test_dataloader, use_gpu = True)
tester.run_link_prediction(type_constrain = False)