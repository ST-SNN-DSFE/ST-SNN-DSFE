from __future__ import division
from __future__ import print_function

import time
import logging
from tqdm import *

import numpy as np
import torch

import torch.optim as optim

from torch.utils.data import TensorDataset, DataLoader
from torch.nn.parameter import Parameter
import pytorch_warmup as warmup

from args import *
from model_ST import ST
from dataloader import load_data_de, load_data_inde
from utils import CE_Label_Smooth_Loss, set_logging_config, save_checkpoint, add_noise
from node_location import convert_dis_m, get_ini_dis_m, return_coordinates
from sklearn.metrics import f1_score

np.set_printoptions(threshold=np.inf)


class Trainer(object):

    def __init__(self, args, subject_name):
        self.args = args
        self.subject_name = subject_name


    def train(self, data_and_label):
        logger = logging.getLogger("train")
        laplacian_array = []
        train_set = TensorDataset((torch.from_numpy(data_and_label["x_tr"])).type(torch.FloatTensor),
                                  (torch.from_numpy(data_and_label["y_tr"])).type(torch.FloatTensor))
        val_set = TensorDataset((torch.from_numpy(data_and_label["x_ts"])).type(torch.FloatTensor),
                                (torch.from_numpy(data_and_label["y_ts"])).type(torch.FloatTensor))

        val_data, val_labels = val_set.tensors
        val_data = add_noise(val_data, noise_type="gaussian")
        val_set = TensorDataset(val_data, val_labels)

        train_loader = DataLoader(train_set, batch_size=self.args.batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_set, batch_size=self.args.batch_size, shuffle=True, drop_last=False)

        adj_matrix = Parameter(torch.FloatTensor(convert_dis_m(get_ini_dis_m(), 9))).to(self.args.device)
        coordinate_matrix = torch.FloatTensor(return_coordinates()).to(self.args.device)

        #####################################################################################
        # 2.define model
        #####################################################################################
        model = ST(self.args, adj_matrix, coordinate_matrix)
        model = model.to(self.args.device)

        lap_params, local_params, weight_params = [], [], []
        for pname, p in model.named_parameters():
        #     print(pname)
            if str(pname) == "adj":
                lap_params += [p]
            elif "local" in str(pname):
                local_params += [p]
            else :
                weight_params += [p]

        optimizer = optim.AdamW([
            {'params': lap_params, 'lr': self.args.beta},
            {'params': local_params, 'lr': self.args.lr},
            {'params': weight_params, 'lr': self.args.lr},
        ], betas=(0.9, 0.999), weight_decay=self.args.weight_decay)

        _loss = CE_Label_Smooth_Loss(classes=self.args.n_class, epsilon=self.args.epsilon).to(self.args.device)

        #############################################################################
        # 3.start train
        #############################################################################
        train_epoch = self.args.epochs

        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                            milestones=[train_epoch // 3],
                                                            gamma=0.1)
        warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)
        warmup_scheduler.last_step = -1

        best_val_acc = 0
        best_val_f1 = 0

        for epoch in range(train_epoch):
            epoch_start_time = time.time()
            train_acc = 0
            train_f1 = 0
            train_loss = 0
            val_loss = 0
            val_f1 = 0
            val_acc = 0

            model.train()
            for i, (x, y) in enumerate(train_loader):
                model.zero_grad()

                x, y = x.to(self.args.device), y.to(device=self.args.device, dtype=torch.int64)
                output, lap_1, _ = model(x)
                loss = _loss(output, y)
                loss.backward()

                optimizer.step()

                if i < len(train_loader) - 1:
                    with warmup_scheduler.dampening():
                        pass

                preds = np.argmax(output.cpu().data.numpy(), axis=1)
                f1 = f1_score(y.cpu().data.numpy(), preds, average='weighted')
                train_f1 += f1 * y.size(0)
                train_acc += np.sum(np.argmax(output.cpu().data.numpy(), axis=1) == y.cpu().data.numpy())
                train_loss += loss.item() * y.size(0)

            train_f1 = train_f1 / train_set.__len__()
            train_acc = train_acc / train_set.__len__()
            train_loss = train_loss / train_set.__len__()

            with warmup_scheduler.dampening():
                lr_scheduler.step()

            model.eval()
            with torch.no_grad():
                for j, (a, b) in enumerate(val_loader):
                    a, b = a.to(self.args.device), b.to(device=self.args.device, dtype=torch.int64)
                    output, lap, fused_feature = model(a)

                    preds = np.argmax(output.cpu().data.numpy(), axis=1)
                    f1 = f1_score(b.cpu().data.numpy(), preds, average='weighted')
                    val_f1 += f1 * b.size(0)
                    val_acc += np.sum(np.argmax(output.cpu().data.numpy(), axis=1) == b.cpu().data.numpy())
                    batch_loss = _loss(output, b)
                    val_loss += batch_loss.item() * b.size(0)

            val_f1 = round(float(val_f1 / val_set.__len__()), 4)
            val_acc = round(float(val_acc / val_set.__len__()), 4)
            val_loss = round(float(val_loss / val_set.__len__()), 4)

            is_best_acc = 0
            if best_val_acc < val_acc:
                best_val_acc = val_acc
                is_best_acc = 1

            if best_val_f1 < val_f1:
                best_val_f1 = val_f1

            if epoch == 0:
                logger.info(self.args)

            if epoch % 5 == 0:
                logger.info("val acc and loss on epoch_{} are: {} and {}".format(epoch, val_acc, val_loss))

            save_checkpoint({
                'iteration': epoch,
                'enc_module_state_dict': model.state_dict(),
                'test_acc': val_acc,
                'optimizer': optimizer.state_dict(),
            }, is_best_acc, self.args.log_dir, self.subject_name)

            if best_val_acc == 1:
                break
        # self.writer.close()
        return best_val_acc, best_val_f1, laplacian_array



def main():
    args = parse_args()
    print("")
    print(f"Current device is {args.device}.")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    logger = logging.getLogger("main")
    logger.info("Logs will only be printed to console, not saved to files.")

    acc_list = []
    acc_dic = {}
    count = 0
    true_path = os.path.join(args.datapath, str(args.session))
    f1_list = []
    for subject in tqdm(os.listdir(true_path)):
        count += 1

        data_and_label = None
        subject_name = str(subject).strip('.npy')
        if args.mode == "dependent":
            logger.info(f"Dependent experiment on {count}th subject : {subject_name}")
            data_and_label = load_data_de(true_path, subject)
        elif args.mode == "independent":
            logger.info(f"Independent experiment on {count}th subject : {subject_name}")
            data_and_label = load_data_inde(true_path, subject)
        else:
            raise ValueError("Wrong mode selected.")

        trainer = Trainer(args, subject_name)
        valAcc, valF1, lap_array = trainer.train(data_and_label)

        acc_list.append(valAcc)
        f1_list.append(valF1)
        lap_array = np.array(lap_array)
        acc_dic[subject_name] = valAcc
        logger.info("Current best acc is : {}".format(acc_dic))
        logger.info("Current average acc is : {}, std is : {}".format(np.mean(acc_list), np.std(acc_list, ddof=1)))
        logger.info("Current average F1 is : {}, std is : {}".format(np.mean(f1_list), np.std(f1_list, ddof=1)))



if __name__ == "__main__":
    main()