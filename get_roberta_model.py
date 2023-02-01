from pre_trained_roberta.get_pretrained_roberta import GetPretrainedRoberta

print('start')
model = GetPretrainedRoberta().get_roberta()
print('moduled: ', model.modules)
model.modules