from random import shuffle
import numpy as np

import torch
from torch.autograd import Variable


class Solver(object):
    default_adam_args = {"lr": 1e-4,
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": 0.0}

    def __init__(self, optim=torch.optim.Adam, optim_args={},
                 loss_func=torch.nn.CrossEntropyLoss()):
        optim_args_merged = self.default_adam_args.copy()
        optim_args_merged.update(optim_args)
        self.optim_args = optim_args_merged
        self.optim = optim
        self.loss_func = loss_func

        self._reset_histories()

    def check_accuracy_dataloader(self,dataloader,model, num_samples = None) :
        """variant of check accuracy that works for more complicated iterators
        Calucates accuracy in batch_sizes specified by the dataloader given as input"""

        batch_size = dataloader.batch_size
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if num_samples is not None:
            print('warning: subsampling in check accuracy not implemented')

        y_pred = []
        acc = 0.
        for batch_nr,(local_batch, local_labels) in enumerate(dataloader):
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)
            scores = model.forward(local_batch)
            y_pred=torch.argmax(scores, 1)
            #print('debug.torch.mean returns : %s'%str(torch.mean((y_pred == local_labels).float())))
            acc =  acc + torch.mean((y_pred == local_labels).float())
        acc = acc / (batch_nr+1.)
        return acc


    def check_accuracy(self, X, y,model, num_samples=None, batch_size=100):
        """
        Check accuracy of the model on the provided data.

        Inputs:
        - X: Array of data, of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,)
        - num_samples: If not None, subsample the data and only test the model
          on num_samples datapoints.
        - batch_size: Split X and y into batches of this size to avoid using too
          much memory.

        Returns:
        - acc: Scalar giving the fraction of instances that were correctly
          classified by the model.
        """
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Maybe subsample the data
        N = X.shape[0]
        if num_samples is not None and N > num_samples:
            mask = np.random.choice(N, num_samples)
            N = num_samples
            X = X[mask].to(device)
            y = y[mask].to(device)

        model.to(device)

        # Compute predictions in batches
        num_batches = N // batch_size
        if N % batch_size != 0:
            num_batches += 1
        y_pred = []

        for i in range(num_batches):
            start = i * batch_size
            end = (i + 1) * batch_size
            scores = model.forward(torch.from_numpy(X[start:end]))
            y_pred.append(torch.argmax(scores, 1))
        y_pred = torch.stack(y_pred,dim=1)
        acc = torch.mean(y_pred == y)

        return acc

    def _reset_histories(self):
        """
        Resets train and val histories for the accuracy and the loss.
        """
        self.train_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        self.val_loss_history = []

    def train(self, model, train_loader, val_loader, num_epochs=10, log_nth=0):
        """
        Train a given model with the provided data.

        Inputs:
        - model: model object initialized from a torch.nn.Module
        - train_loader: train data in torch.utils.data.DataLoader
        - val_loader: val data in torch.utils.data.DataLoader
        - num_epochs: total number of training epochs
        - log_nth: log training accuracy and loss every nth iteration
        """
        optim = self.optim(model.parameters(), **self.optim_args)
        self._reset_histories()
        iter_per_epoch = len(train_loader)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

        print('START TRAIN.')

        for epoch in range(num_epochs):
            for batch_nr, (local_batch, local_labels) in enumerate(train_loader):

                local_batch, local_labels = local_batch.to(device), local_labels.to(device)
                optim.zero_grad()  # zero the gradient buffers
                output = model.forward(local_batch)
                loss = self.loss_func(output, local_labels)
                loss.backward()
                optim.step()  # Does the update based on the accumalted gradients
                #if batch_nr%100 == 0 :
                #    print("[Iteration %d/%d] TRAIN loss: %f "%((batch_nr+1.)*train_loader.batch_size,len(val_loader)+1,loss))


            #now append training loss and validation acc at end of each epoch
            train_acc = self.check_accuracy_dataloader(train_loader,model,
                                            )
            val_acc = self.check_accuracy_dataloader(val_loader,model)
            self.train_acc_history.append(train_acc)
            self.val_acc_history.append(val_acc)
            print("[Epoch %d/%d] TRAIN acc: %f"%(epoch+1,num_epochs+1,train_acc))
            print("[Epoch %d/%d] VAL   acc: %f" % (epoch+1,num_epochs+1,val_acc))


        print('FINISH.')
