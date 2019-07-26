'''
数据集划分
'''

import os
import random
expressions = ['happiness', 'disgust', 'repression', 'surprise', 'others']
train_proportion = 4 / 5


CASMEII_root = '/media/zxl/other/pjh/datasetsss/CASME_II/'
f1 = open('./' + 'train_list' + '.txt', 'w')
f2 = open('./' + 'test_list' + '.txt', 'w')
for label, expression in enumerate(expressions):
    expression_path = CASMEII_root + expression
    samples = os.listdir(expression_path)
    samples.sort(key=lambda x: int(x))

    train_samples = random.sample(range(len(samples)), int(len(samples)*train_proportion))
    for sample in samples:
        if int(sample) in train_samples:
            f1.write(expression + '/' + sample + ' ' + str(label) + '\n')
        else:
            f2.write(expression + '/' + sample + ' ' + str(label) + '\n')
print('-'*72)
print('-->Train and test samples have been updated in train_list and test_list!')
print('-'*72)

