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
from openke.module.model import TransE
from openke.module.loss import MarginLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader

from utils_zoe import mkdir

mkdir('./checkpoint')

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
transe = TransE(
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
	dim = 200, 
	p_norm = 1, 
	norm_flag = True)


# define the loss function
model = NegativeSampling(
	model = transe, 
	loss = MarginLoss(margin = 5.0),
	batch_size = train_dataloader.get_batch_size()
)

# train the model
trainer = Trainer(model = model, data_loader = train_dataloader, train_times = 1000, alpha = 1.0, use_gpu = True)
trainer.run()
mkdir
transe.save_checkpoint('./checkpoint/transe.ckpt')

# test the model
transe.load_checkpoint('./checkpoint/transe.ckpt')
tester = Tester(model = transe, data_loader = test_dataloader, use_gpu = True)
tester.run_link_prediction(type_constrain = False)