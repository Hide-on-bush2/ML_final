from __future__ import absolute_import, division, print_function, unicode_literals
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np


if __name__ == "__main__":

	# 获取数据集

	# df = pd.read_csv("../data/titanic/clean_trainset.csv")
	# print(df)

	column_name = ['Survived', 'Pclass', 'Age', 'Fare', 'familySize', 'TickGroup',
       'Sex_female', 'Sex_male', 'Embarked_C', 'Embarked_Q', 'Embarked_S',
       'Title_Master', 'Title_Miss', 'Title_Mr', 'Title_Mrs', 'Title_Officer',
       'Title_Royalty', 'Deck_A', 'Deck_B', 'Deck_C', 'Deck_D', 'Deck_E',
       'Deck_F', 'Deck_G', 'Deck_T', 'Deck_U', 'Pclass_1', 'Pclass_2',
       'Pclass_3', 'TickGroup_0', 'TickGroup_1', 'TickGroup_2', 'familySize_0',
       'familySize_1', 'familySize_2']
	feature_name = column_name[1:]
	label_name = column_name[0]
	# column_defaults = [tf.int32, tf.float32, tf.float32, tf.int32, tf.int32, 
 #    	tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, 
 #    	tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, 
 #    	tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, 
 #    	tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, 
 #    	tf.int32]
        

	# print("Features: {}".format(feature_name))
	# print("label: {}".format(label_name))

	class_name = ["die", "live"]

	# 创建可以用来训练的数据集

	batch_size = 32

	train_dataset = tf.data.experimental.make_csv_dataset(
		"../data/titanic/clean_trainset.csv",
		batch_size,
		column_names=column_name,
		label_name=label_name,
		num_epochs=1)

	features, labels = next(iter(train_dataset))
	# print(features)

	def pack_features_vector(features, labels):
		"""将特征打包到一个数组中"""
		features = tf.stack(list(features.values()), axis=1)
		return features, labels

	train_dataset = train_dataset.map(pack_features_vector)

	features, labels = next(iter(train_dataset))
	# print(features[0:5])

	# 创建模型
	model = tf.keras.Sequential([
		tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(34, )),
		tf.keras.layers.Dense(10, activation=tf.nn.relu),
		tf.keras.layers.Dense(3)
		])

	# 定义损失函数

	loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

	def loss(model, x, y):
		y_ = model(x)
		return loss_object(y_true=y, y_pred=y_)

	l = loss(model, features, labels)

	print("Loss Test: {}".format(l))

	# 计算梯度

	def grad(model, inputs, targets):
		with tf.GradientTape() as tape:
			loss_value = loss(model, inputs, targets)
		return loss_value, tape.gradient(loss_value, model.trainable_variables)

	# 设置优化器

	optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

	# 保留结果用于绘制图表

	train_loss_results = []
	train_accuracy_results = []

	# 开始训练

	num_epochs = 201

	for epoch in range(num_epochs):
		epoch_loss_avg = tf.keras.metrics.Mean()
		epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

		# 训练循环-使用大小为32的batchs
		for x, y in train_dataset:
			# 优化模型
			loss_value, grads = grad(model, x, y)
			optimizer.apply_gradients(zip(grads, model.trainable_variables))

			# 追踪进度
			epoch_loss_avg(loss_value) # 添加当前的batch loss
			epoch_accuracy(y, model(x)) # 比较预测标签和真实标签

		# 循环结束
		train_loss_results.append(epoch_loss_avg.result())
		train_accuracy_results.append(epoch_accuracy.result())

		if epoch % 50 == 0:
			print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
																		epoch_loss_avg.result(),
																		epoch_accuracy.result()))


	# 绘制可视化损失函数随时间推移而变化的情况

	# fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
	# fig.suptitle('Training Metrics')

	# axes[0].set_ylabel("Loss", fontsize=14)
	# axes[0].plot(train_loss_results)

	# axes[1].set_ylabel("Accuracy", fontsize=14)
	# axes[1].set_xlabel("Epoch", fontsize=14)
	# axes[1].plot(train_accuracy_results)
	# plt.show()


	# 建立测试数据集

	test_dataset = pd.read_csv("../data/titanic/clean_testset.csv")

	predict_dataset = tf.convert_to_tensor(np.array(test_dataset[feature_name]))

	predictions = model(predict_dataset)

	test_label = []

	for i, logits in enumerate(predictions):
		class_idx = tf.argmax(logits).numpy()
		p = tf.nn.softmax(logits)[class_idx]
		name = class_name[class_idx]
		print("Example {} prediction: {} ({:4.1f}%)".format(i, name, 100*p))
		test_label.append(class_idx)


	# 建立一个保存预测结果的df
	test_dataset['PassengerId'] = full_set['PassengerId'][full_set['Survived'].isnull()]

	pre_saver = pd.DataFrame({'PassengerId':test_dataset['PassengerId'], 'Survived':test_label})
	pre_saver.to_csv("../data/titanic/predictions.csv", index=False)
	print(test_label)
