环境需求：
	NVIDIA GTX 1080 Ti
	CUDA 8.0
	Python 3.5.3
	Pytorch 0.4.0
	torchvision  0.2.1
	opencv 3.2.0
Data（256*256）
	src_data(来自真实世界的图片)
			train
			test
	tgt_data
			train（动漫图片）
			pair(对动漫图片进行边缘模糊化处理后）
	output_image           测试结果图片
	test                           测试图片

若要重新训练，请不要创建pair文件夹，它会在我们执行训练时（即python CartoonGAN.py）自动创建

训练模型：
python CartoonGAN.py --name pytorch-CartoonGAN-master --src_data src_data --tgt_data tgt_data --vgg_model pretrained_model/vgg19.pth

这里首先会进行 edge_promoting 进行边缘模糊处理，然后再进行预训练，最后进行生成对抗网络的训练

测试模型：
python test.py --pre_trained_mode project_name_results/generator_param.pkl --image_dir data/  --output_image_dir data/output_image

若重新训练后需要测试：
首先在data文件夹下创建test文件夹用于放置你要测试的图片，并把你要测试的图片放入其中；接着创建一个output_image文件夹，用于保存最终输出的结果

若重新训练 project_name_results 这个文件夹会自动生成

我们运行完的结果保存在data里