import matplotlib.pyplot as plt
import torch
import numpy
from sklearn.metrics import f1_score
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
 
class RobertaHandler:
    def __init__ (self):
        pass
    
    def estimate_roberta_model_accuracy(self, model_outputs, labels):
        y_preds = torch.zeros(torch.tensor(model_outputs).view(-1,).shape[0], dtype=torch.int64)
        # print('model_outputs shape: ', torch.tensor(model_outputs).view(-1,))
        # print('y_preds output shape: ', y_preds)
        model_preds = torch.tensor(model_outputs).view(-1,) > 0.5
        y_preds[model_preds] = 1
        # y_preds = torch.tensor(pd.Series(y_preds.detach())).view(-1,)
        
        # print('y pred est:', y_preds)
        # print('labels est:', labels)

        return f1_score(y_pred=y_preds, y_true=labels)

    def get_roberta_train_data_accuracy(self, roberta_model, data, batch_size = 1):
        model = roberta_model.model
        tokenizer = roberta_model.tokenizer
        model.eval()
        model_outputs = []
        for i in range(0, len(data), batch_size):
            if (i + batch_size) > len(data):
                break
            data_idx = range(i,i+batch_size)
            text_data = data['text'][data.index[data_idx]].to_list()
            preprocessed_data = tokenizer(text_data, return_tensors='pt', padding=True).to(device)

            model_output = model.forward(**preprocessed_data).logits
            preprocessed_data.to('cpu')
            # print('model output shape: ', model_output, ' and after view: ',model_output.view(-1,).detach())
            model_outputs.append(model_output.view(-1,).detach().to('cpu').tolist())
        model_train_labels = data['target']
        return self.estimate_roberta_model_accuracy(model_outputs, model_train_labels)



    def train_roberta_model(self, model, train_data, val_data, learning_rate=1e-3, weight_decay=0, batch_size=32, num_epoches=30, check_point_path=None):
        
        roberta_model = model.model
        tokenizer = model.tokenizer
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(roberta_model.parameters(),
                                lr=learning_rate,
                                weight_decay=weight_decay)

        epoches, train_losses, val_losses = [], [], []
        train_accs, val_accs  = [], []
        
        for epoch in range(0,num_epoches):
            roberta_model.train()
            model_val_outputs = []
            epoch_train_loss = 0
            train_iterations_counter = 0
            

            train_random_order = torch.randperm(len(train_data)).numpy()
            val_random_order = torch.randperm(len(val_data)).numpy()
            for i in range(0, len(train_data), batch_size):
                if (i + batch_size) > len(train_data):
                    break
                train_iterations_counter += 1
                data_idx = train_random_order[i:i+batch_size]
                labels = train_data['target'][train_data.index[data_idx]].to_list()
                labels = torch.tensor([labels], dtype=torch.float32).view(batch_size, 1).to(device)
                data = train_data['text'][train_data.index[data_idx]].to_list()
                preprocessed_data = tokenizer(data, return_tensors='pt', padding=True).to(device)
                # print(preprocessed_data)
                # print('labels on train: ', labels, ' labels shape: ', labels.shape)
                model_output = roberta_model.forward(**preprocessed_data).logits
                # print('model_output on train: ', model_output)
                loss = criterion(model_output, labels)                   # compute the total loss
                loss.backward()                      # compute updates for each parameter
                optimizer.step()                      # make the updates for each parameter
                optimizer.zero_grad() 
                preprocessed_data.to('cpu')
                epoch_train_loss += loss.item()/batch_size
            train_losses.append(epoch_train_loss/train_iterations_counter)
            train_accs.append(self.get_roberta_train_data_accuracy(model, train_data, batch_size))

            roberta_model.eval()
            val_iterations_counter = 0
            epoch_val_loss = 0
            for i in range(0, len(val_data), batch_size):
                if (i + batch_size) > len(val_data):
                    break
                val_iterations_counter += 1
                data_idx = range(i,i+batch_size)
                labels = val_data['target'][val_data.index[data_idx]].to_list()
                labels = torch.tensor([labels], dtype=torch.float32).view(batch_size, 1).to(device)
                data = val_data['text'][val_data.index[data_idx]].to_list()
                preprocessed_data = tokenizer(data, return_tensors='pt', padding=True).to(device)

                model_output = roberta_model.forward(**preprocessed_data).logits
                loss = criterion(model_output, labels)                   # compute the total loss

                preprocessed_data.to('cpu')
                labels.to('cpu')
                model_val_outputs.append(model_output.view(-1,).detach().to('cpu').tolist())
                
                # print(model_output.logits.detach().numpy())
                epoch_val_loss += loss.item()/batch_size

            model_val_labels = val_data['target']
            val_losses.append(epoch_val_loss/ val_iterations_counter)
            epoches.append(epoch)
            val_acc_estimation = self.estimate_roberta_model_accuracy(model_val_outputs, model_val_labels)
            val_accs.append(val_acc_estimation)
            print(f"====> Epoch: {epoch} Average train loss: {train_losses[epoch]:.4f}, train acc: {train_accs[epoch]:.4f}, Average val loss: {val_losses[epoch]:.4f}, val acc: {val_accs[epoch]:.4f}")
        
        return epoches, train_losses, val_losses, train_accs, val_accs

    def plot_learning_curve(epoches, train_losses, val_losses, train_accs, val_accs):
        """
        Plot the learning curve.
        """
        plt.title("Learning Curve: Loss per epoch")
        plt.plot(epoches, train_losses, label="Train")
        plt.plot(epoches, val_losses, label="Validation")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.legend(['Train', 'Train'])
        plt.show()

        plt.title("Learning Curve: Accuracy per epoch")
        plt.plot(epoches, train_accs, label="Train")
        plt.plot(epoches, val_accs, label="Validation")
        plt.xlabel("Iterations")
        plt.ylabel("Accuracy")
        plt.legend(['Train', 'Train'])
        plt.show()