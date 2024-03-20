import logging

logging.basicConfig(
    level=logging.INFO,
    format="",
    handlers=[
        logging.StreamHandler()
    ]
)

import os
import sys
import torch
import datetime
import numpy as np
import framework.Metrics as metrics
from torch.utils.data import DataLoader

try:
    from torch.utils.tensorboard.writer import SummaryWriter
except ImportError:
    logging.info("You have elected not to install tensorboard")

class SupervisedMLFramework:
    def __init__(self, model, model_name="", out_dir=".", train_dataset=None, test_dataset=None, custom_validation_dataset=None) -> None:
        self.model = model
        self.model_name = model_name
        self.out_dir = os.path.join(out_dir, model_name)
        self.run_dir = os.path.join(self.out_dir, "runs")
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.custom_validation_dataset = custom_validation_dataset

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Using {self.device}")

        #Move model to GPU if available
        self.model = model.to(self.device)

        if "tensorboard" in sys.modules:
            self.writer = SummaryWriter()
        else:
            self.writer = None
        if not os.path.exists(os.path.join(self.out_dir)):
            os.makedirs(os.path.join(self.out_dir))

    def __del__(self):
        if self.writer != None:
            self.writer.close()

    def test(self, loss_function, batch_size):

        run_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        run_path = os.path.join(self.run_dir, run_time)

        if not os.path.exists(run_path):
            os.makedirs(run_path)

        logger =  logging.getLogger("test_logger")
        logger.addHandler(logging.FileHandler(os.path.join(run_path, f"{run_time}.log")))

        logger.info(f"\n\n{'*'*10} Evaluating on test set {'*'*10}\n\n")
        self.log_info(logger, loss_function=loss_function, epochs=1, batch_size=batch_size)

        test_dataloader = DataLoader(self.test_dataset, batch_size, shuffle=False)

        testing_loss, correct = 0, 0
        self.model.eval()

        labels = []
        predictions = []

        with torch.no_grad():
            for batch, (X, y) in enumerate(test_dataloader):
                labels += y.tolist()

                X = X.to(self.device)
                y = y.to(self.device)

                prediction = self.model(X)
                loss = loss_function(prediction, y)

                test = prediction.to('cpu').argmax(1).tolist()
                predictions += test

                correct += (prediction.argmax(1) == y).type(torch.float).sum().item()
                testing_loss += loss.item()

                if batch % 10 == 0:
                    current = (batch + 1) * len(X)
                    logger.info(f"batch {batch} loss: {loss}  [{current:>5d}/{len(test_dataloader) * batch_size:>5d}]")

        logger.info("\n")
        logger.info(f"Percentage correct: {correct / len(self.test_dataset) * 100}")
        logger.info(f"Average testing loss: {testing_loss / len(test_dataloader)}")

    def train(self,  epochs, loss_function, optim, optim_params, batch_size, sched=None, sched_params=None, weight_save_period=5, patience=3, use_custom_validation_set=False, validation_percent=20):

        run_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        run_path = os.path.join(self.run_dir, run_time)

        if not os.path.exists(run_path):
            os.makedirs(run_path)
        if os.path.isfile(os.path.join(run_path, "train_losses.txt")):
            open(os.path.join(run_path, "train_losses.txt"), "w").close() # clear out the file

        logger =  logging.getLogger("train_logger")
        logger.addHandler(logging.FileHandler(os.path.join(run_path, f"{run_time}.log")))

        optimizer = optim(self.model.parameters(), **optim_params)

        if sched != None:
            scheduler = sched(optimizer, **sched_params)

        logger.info(f"\n\n{'*'*10} Running standard training {'*'*10}\n\n")
        self.log_info(logger, epochs, loss_function, optimizer, batch_size, patience, weight_save_period)

        # Construct train and validation dataloaders
        if use_custom_validation_set:

            if self.custom_validation_dataset == None:
                logging.error("No custom validation set provided to SupervisedMLFramework constructor")
                return

            validation_dataloader = DataLoader(self.custom_validation_dataset, batch_size, shuffle=False)
            train_dataloader = DataLoader(self.train_dataset, batch_size, shuffle=True)

        else:
            indices = np.random.permutation(len(self.train_dataset))
            cutoff_index = int(len(self.train_dataset) * validation_percent / 100)
            validation_indices = indices[:cutoff_index]
            train_indices = indices[cutoff_index:]

            validation_subdataset = torch.utils.data.Subset(self.train_dataset, indices[validation_indices])
            train_subdataset = torch.utils.data.Subset(self.train_dataset, indices[train_indices])

            validation_dataloader = DataLoader(validation_subdataset, batch_size, shuffle=False)
            train_dataloader = DataLoader(train_subdataset, batch_size, shuffle=True)

        for epoch in range(epochs):
            logger.info(f"\n {'-'*10} Epoch {epoch} {'-'*10} ")
            total_train_loss_batch = 0
            num_correct = 0
            for batch, (X, y) in enumerate(train_dataloader):
                
                X = X.to(self.device)
                y = y.to(self.device)

                optimizer.zero_grad()
                prediction = self.model(X)
                loss = loss_function(prediction, y)

                loss.backward()
                optimizer.step()
                
                num_correct += (prediction.argmax(1) == y).type(torch.float).sum().item()

                avg_loss_over_batch = loss.item()
                total_train_loss_batch += avg_loss_over_batch

                if batch % 10 == 0:
                    current = (batch + 1) * len(X)
                    with open(os.path.join(run_path, f"train_losses.txt"), "a") as f:
                        f.write(f"epoch {epoch}, batch_number_accumulated {epoch * len(train_dataloader) + batch}, loss_over_batch {avg_loss_over_batch}\n")
                    logger.info(f"batch {batch} loss: {loss}  [{current:>5d}/{len(train_dataloader) * batch_size:>5d}]")

            average_epoch_train_loss = total_train_loss_batch / len(train_dataloader)
            logger.info(f"Average loss over entire epoch: {average_epoch_train_loss}")
            logger.info(f"Percent accuracy over epoch: {num_correct / len(train_dataloader.dataset) * 100}")

            torch.cuda.empty_cache()

            if self.writer != None:
                self.writer.add_scalar('Loss/train', average_epoch_train_loss, epoch)

            logger.info(f"\n {'-'*10} Validating (epoch {epoch}) {'-'*10} ")
            labels = []
            predictions = []
            epoch_validation_loss = 0
            num_correct = 0
            for batch, (X, y) in enumerate(validation_dataloader):
                with torch.no_grad():
                    X = X.to(self.device)
                    y = y.to(self.device)
                    
                    # prediction is expected to be unnormalized logits
                    prediction = self.model(X)
                    loss = loss_function(prediction, y)

                    labels.extend(y.tolist())
                    predictions.extend(prediction.argmax(1).tolist())

                    num_correct += (prediction.argmax(1) == y).type(torch.float).sum().item()
                    loss = loss.item()
                    epoch_validation_loss += loss

                    if batch % 10 == 0:
                        current = (batch + 1) * len(X)
                        with open(os.path.join(run_path, f"validation_losses.txt"), "a") as f:
                            f.write(f"epoch {epoch}, batch_number_accumulated {epoch * len(validation_dataloader) + batch}, loss_over_batch {loss}\n")
                        logger.info(f"batch {batch} loss: {loss}  [{current:>5d}/{len(validation_dataloader)* batch_size:>5d}]")

            avg_epoch_validation_loss = epoch_validation_loss / len(validation_dataloader)

            logger.info(metrics.print_metrics(np.array(predictions), np.array(labels)))

            logger.info(f"Average batch validation loss for epoch {epoch}: {avg_epoch_validation_loss}")

            if sched != None:
                scheduler.step()

            if self.writer != None:
                self.writer.add_scalar('Loss/validation', avg_epoch_validation_loss, epoch)

            if epoch == 0:
                min_loss = avg_epoch_validation_loss
                self.stop_counter = 0
                continue

            logger.info(f"Previous min loss: {min_loss}\nCurrent Loss: {avg_epoch_validation_loss}\n")
            #if validation loss at a min save it
            if (avg_epoch_validation_loss < min_loss):
                self.stop_counter = 0
                min_loss = avg_epoch_validation_loss

                logger.info("Saving weights: validation loss is at minimum\n")
                logger.info(f"New min loss: {min_loss}")

                if epoch % weight_save_period == 0:
                    save_path = os.path.join(run_path,f"epoch_{epoch}_weights.pt")
                    torch.save(self.model.state_dict(), save_path)
                else:
                    #if not divisible by weight_save_period then overwrite weights for the previous number. We don't need to keep intermediates
                    save_path =  os.path.join(run_path,f"epoch_{int(epoch / weight_save_period) * weight_save_period}_weights.pt")
                    torch.save(self.model.state_dict(), save_path)

                logger.info(f"Weights saved to {save_path}\n\n")

            else:
                logger.info("epoch loss failed to decrease\n")
                if self.stop_counter >= patience:
                    self.stop_counter = 0
                    break
                else:
                    self.stop_counter += 1
                    logging.info(f"stop counter: {self.stop_counter} / {patience}\n\n")
                    continue


    def tune(self, epochs, loss_function, optim, optim_params, batch_size, sched=None, sched_params=None, k=4, weight_save_period=5, patience=20, use_custom_validation_set=False):

        run_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        run_path = os.path.join(self.run_dir, run_time)

        if not os.path.exists(run_path):
            os.makedirs(run_path)
        if os.path.isfile(os.path.join(run_path, "train_losses.txt")):
            open(os.path.join(run_path, "train_losses.txt"), "w").close() #clear out the file

        logger =  logging.getLogger("tune_logger")
        logger.addHandler(logging.FileHandler(os.path.join(run_path, f"{run_time}.log")))

        logger.info(f"\n\n{'*'*10} Running {k}-fold cross validation {'*'*10}\n\n")

        indices = np.random.permutation(range(len(self.train_dataset)))
        fold_size = int(len(indices) / k)
        k_fold_indices = list(range(len(indices)))

        split_validation_losses = []
        split_percentage_corrects = []
        min_loss = 1.0
        for split in range(1, k + 1):
            logger.info(f"\n\n{'*'*10}Training split {split} / {k}{'*'*10}")
            self.stop_counter = 0

            optimizer = optim(self.model.parameters(), **optim_params)

            self.log_info(logger, epochs, loss_function, optimizer, batch_size, patience, weight_save_period)

            if sched != None:
                scheduler = sched(optimizer, **sched_params)

            if (split > 1):
                logger.info("\n\n\nResetting model\n\n\n")
                self.model.reset()
                self.model = self.model.to(self.device)


            if use_custom_validation_set:
               if self.custom_validation_dataset == None:
                    logging.error("No custom validation set provided to SupervisedMLFramework constructor")
                    return

               validation_dataloader = DataLoader(self.custom_validation_dataset, batch_size, shuffle=False)
               train_dataloader = DataLoader(self.train_dataset, batch_size, shuffle=True)
            else:
                validation_indices = k_fold_indices[(split -1) *fold_size: split *fold_size] if split != k else k_fold_indices[(split -1) *fold_size:]
                train_indices = list(set(k_fold_indices)  - set(validation_indices))

                validation_subdataset = torch.utils.data.Subset(self.train_dataset, indices[validation_indices])
                train_subdataset = torch.utils.data.Subset(self.train_dataset, indices[train_indices])

                validation_dataloader = DataLoader(validation_subdataset, batch_size, shuffle=True)
                train_dataloader = DataLoader(train_subdataset, batch_size, shuffle=True)

            split_validation_loss = 0
            split_percentage_correct = 0

            for epoch in range(epochs):
                logger.info(f"\n {'-'*10} Epoch {epoch} {'-'*10} ")
                total_train_loss_batch = 0
                for batch, (X, y) in enumerate(train_dataloader):
                    X = X.to(self.device)
                    y = y.to(self.device)

                    optimizer.zero_grad()
                    prediction = self.model(X)
                    loss = loss_function(prediction, y)

                    loss.backward()
                    optimizer.step()

                    avg_loss_over_batch = loss.item()
                    total_train_loss_batch += avg_loss_over_batch

                    if batch % 10 == 0:
                        current = (batch + 1) * len(X)
                        with open(os.path.join(run_path, f"split_{split}_train_losses.txt"), "a") as f:
                            f.write(f"epoch {epoch}, batch_number_accumulated {epoch * len(train_dataloader) + batch}, loss_over_batch {avg_loss_over_batch}\n")
                        logger.info(f"batch {batch} loss: {loss}  [{current:>5d}/{len(train_dataloader) * batch_size:>5d}]")

                average_epoch_train_loss = total_train_loss_batch / len(train_dataloader)
                logger.info(f"Average loss over entire epoch: {average_epoch_train_loss}")

                torch.cuda.empty_cache()

                if self.writer != None:
                    self.writer.add_scalar('Loss/train', average_epoch_train_loss, epoch)

                logger.info(f"\n {'-'*10} Validating (epoch {epoch}) {'-'*10} ")
                predictions = []
                labels = []
                epoch_validation_loss = 0
                num_correct = 0
                for batch, (X, y) in enumerate(validation_dataloader):
                    with torch.no_grad():
                        X = X.to(self.device)
                        y = y.to(self.device)

                        # prediction is expected to be unnormalized logits
                        prediction = self.model(X)
                        loss = loss_function(prediction, y)

                        predictions.extend(prediction.argmax(1).tolist())
                        labels.extend(y.tolist())

                        num_correct += (prediction.argmax(1) == y).type(torch.float).sum().item()
                        loss = loss.item()
                        epoch_validation_loss += loss

                        if batch % 10 == 0:
                            current = (batch + 1) * len(X)
                            with open(os.path.join(run_path, f"split_{split}_validation_losses.txt"), "a") as f:
                                f.write(f"epoch {epoch}, batch_number_accumulated {epoch * len(validation_dataloader) + batch}, loss_over_batch {loss}\n")
                                logger.info(f"batch {batch} loss: {loss}  [{current:>5d}/{len(validation_dataloader) * batch_size:>5d}]")

                avg_epoch_validation_loss = epoch_validation_loss / len(validation_dataloader)
                split_validation_loss += avg_epoch_validation_loss
                split_percentage_correct += num_correct / (len(validation_dataloader) * batch_size) * 100

                logger.info(metrics.print_metrics(np.array(predictions), np.array(labels)))

                logger.info(f"Average batch validation loss for epoch {epoch}: {avg_epoch_validation_loss}")

                if sched != None:
                    scheduler.step()

                if self.writer != None:
                    self.writer.add_scalar('Loss/validation', avg_epoch_validation_loss, epoch)

                if epoch == 0:
                    min_loss = avg_epoch_validation_loss
                    self.stop_counter = 0
                    continue

                logger.info("--Previous min loss--")
                logger.info(min_loss)
                #if validation loss at a min save it
                if (avg_epoch_validation_loss < min_loss):
                    self.stop_counter = 0
                    min_loss = avg_epoch_validation_loss

                    logger.info("Saving weights: validation loss is at minimum\n")
                    logger.info(f"New min loss:{min_loss}")

                    if epoch % weight_save_period == 0:
                        torch.save(self.model.state_dict(), os.path.join(run_path,f"split_{split}_epoch_{epoch}_weights.pt"))
                    else:
                        #if not divisible by weight_save_period then overwrite weights for the previous number. We don't need to keep intermediates
                        torch.save(self.model.state_dict(), os.path.join(run_path,f"split_{split}_epoch_{int(epoch / weight_save_period) * weight_save_period}_weights.pt"))
                else:
                    logger.info("epoch loss failed to decrease\n")
                    if self.stop_counter >= patience:
                        self.stop_counter = 0
                        break
                    else:
                        self.stop_counter += 1
                        logging.info(f"stop counter: {self.stop_counter} / {patience}")
                        continue

            logger.info(f"Average validation loss over all epochs: {split_validation_loss / (epochs)}\n")
            split_validation_losses.append(split_validation_loss / (epochs))
            logger.info(f"Average validation accuracy over all epochs: {split_percentage_correct / (epochs)} %\n")
            split_percentage_corrects.append(split_percentage_correct / (epochs))

        logger.info(f"{'*'*10}Accumulated Metrics{'*'*10}")
        logger.info(f"Mean validation losses per split: {split_validation_losses}")
        avg_validation_loss_all_splits = sum(split_validation_losses) / len(split_validation_losses)
        logger.info(f"Average validation loss over all splits: {avg_validation_loss_all_splits}")

        logger.info(f"Mean validation accuracies per split: {split_percentage_corrects}")
        avg_validation_accuracy_all_splits = sum(split_percentage_corrects) / len(split_percentage_corrects)
        logger.info(f"Average validation accuracy over all splits: {avg_validation_accuracy_all_splits} %\n\n")

        logger.removeHandler(logger.handlers[0])

    def predict(self, sample, transform):
        sample = sample.to(self.device)
        sample = transform(sample)
        self.model.eval()
        with torch.no_grad():
            return self.model(sample)

    def log_info(self, logger=logging.getLogger(), epochs=None, loss_function=None, optimizer=None, batch_size=None, patience=None, weight_save_period=None):

        output_str = "\n".join((
                f"{'-' * 30}",
                f"* Model Name: {self.model_name}",
                f"* Number of Epochs: {epochs}",
                f"* Loss Function: {loss_function}",
                f"* Optimizer: {optimizer}",
                f"* Batch Size: {batch_size}",
                f"* Patience: {patience}",
                f"* Weight Save Period: {weight_save_period}",
                f"{'-' * 30}",
        ))

        logger.info("Parameters")
        logger.info(output_str)

