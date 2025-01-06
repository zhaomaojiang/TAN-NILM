"""For the train program, all electrical appliances corresponding to REFIT have a python file of train and test"""
import os
import sys
from data.train_loader import DataGenerator, save_checkpoint
import torch
from tqdm import tqdm
from nnet.tans import TAN_Plus, compute_loss, compute_loss_val, compute_ce_loss  # 训练用MSE，测试用MAE（刚好找最好）

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
current_dir = os.path.dirname(os.path.abspath('__file__'))
root_dir = os.path.abspath(os.path.join(current_dir, '../../../'))
sys.path.append(root_dir)
direction = './data/REFIT_dataset/refit_64dis_sam8_sen0'  # todo:

data_set_name = 'REFIT'
app_name = 'dishwasher'  # todo: 'washingmachine', 'dishwasher', 'microwave', 'fridge', 'kettle'
if data_set_name == 'REDD':
    batch_si = 32
    all_ap = ['dishwasher', 'microwave', 'fridge']
else:
    batch_si = 64
    all_ap = ['washingmachine', 'dishwasher', 'microwave', 'fridge', 'kettle']

check_path = 'check_' + app_name + '/'
val_losses = {}
all_losses = []
# todo:
model = TAN_Plus(
    L1=3,
    N=32,
    B=3,
    O=32,
    P=96,
    Q=3,
    num_spks=5,
    spk_embed_dim=32,
    Head=4,
    causal=False).cuda()

numparams = 0
for f in model.parameters():
    if f.requires_grad:
        numparams += f.numel()
print('Trainable Parameters: {}'.format(numparams))

dataset = DataGenerator(direction,
                        batch_size=batch_si,
                        app=all_ap,
                        threshold=False,
                        norm=False)

batch_num = dataset.tr_num
batch_num_te = dataset.te_num
batch_num_val = dataset.va_num


def run(check='check_' + app_name + '/spexlast.pt', lr=1e-4, e_num=100):
    best_val_loss = 10000
    clip_grad_norm = 0
    tr_step = 0
    loss_tr_list, loss_val_list, loss_te_list = [], [], []
    ce_tr_list, ce_val_list, ce_te_list = [], [], []
    loss_val_sep = {}
    epoch = 0
    no_improve_epochs = 0
    patience = 30
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    if check != 0:
        checkpoint = torch.load(check)
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint['epoch']
        no_improve_epochs = checkpoint['no_improve_epochs']
        tr_step = epoch + 1
        loss_tr_list, loss_val_list = checkpoint['mae_loss_list']
        ce_tr_list, ce_val_list = checkpoint['ce_loss_list']
        best_val_loss = checkpoint['best_para'][0]
        opt.load_state_dict(checkpoint['optim_state_dict'])

    for i in range(e_num):
        ce_val_sum, ce_te_sum, ce_tr_sum = 0, 0, 0
        maeloss_val_sum, maeloss_te_sum, mseloss_tr_sum = 0, 0, 0
        for key in dataset.app:
            loss_val_sep[key] = 0
        model.train()
        for _ in tqdm(range(batch_num), desc='Training'):
            opt.zero_grad()
            data = next(dataset.gen_tr())
            if len(data) != 5:
                aux = torch.tensor(data[0], dtype=torch.float32).cuda()
                aux_len = torch.tensor(data[1]).cuda()
                class_label = torch.tensor(data[2]).long().cuda()
                pred_out = model.predce(aux, aux_len)
                ce_loss_tr = compute_ce_loss(pred_out, class_label)
                mseloss_tr_sum += 0
                ce_tr_sum += ce_loss_tr
                loss_tr = 5 * ce_loss_tr
            else:
                mix_wavs = torch.tensor(data[0].astype(float), dtype=torch.float32).cuda()
                clean_wavs = torch.tensor(data[1].astype(float), dtype=torch.float32).cuda()
                aux = torch.tensor(data[2], dtype=torch.float32).cuda()
                aux_len = torch.tensor(data[3]).cuda()
                class_label = torch.tensor(data[4]).long().cuda()
                rec_app_wav = model(mix_wavs, aux, aux_len)
                mae_loss_tr, ce_loss_tr = compute_loss(rec_app_wav, clean_wavs, class_label)
                mseloss_tr_sum += mae_loss_tr
                ce_tr_sum += ce_loss_tr
                loss_tr = mae_loss_tr + 5 * ce_loss_tr
            loss_tr.backward()

            if clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               clip_grad_norm)
            opt.step()
            tr_step += 1
        del mix_wavs, clean_wavs, aux, aux_len, class_label, rec_app_wav, data

        model.eval()
        with torch.no_grad():
            for _ in tqdm(range(batch_num_val), desc='Validation{}'):
                val = next(dataset.gen_val())
                if len(val) != 5:
                    aux = torch.tensor(val[0], dtype=torch.float32).cuda()
                    aux_len = torch.tensor(val[1]).cuda()
                    one_hot = torch.tensor(val[2]).long().cuda()
                    pred_out = model.predce(aux, aux_len)
                    ce_loss_val = compute_ce_loss(pred_out, one_hot)
                    maeloss_val_sum += 0
                    ce_val_sum += ce_loss_val
                else:
                    mix_wavs = torch.tensor(val[0].astype(float), dtype=torch.float32).cuda()
                    clean_wavs = torch.tensor(val[1].astype(float), dtype=torch.float32).cuda()
                    aux = torch.tensor(val[2], dtype=torch.float32).cuda()
                    aux_len = torch.tensor(val[3]).cuda()
                    one_hot = torch.tensor(val[4]).long().cuda()
                    rec_app_wav = model(mix_wavs, aux, aux_len)
                    l1_loss_val, ce_loss_val = compute_loss_val(rec_app_wav,
                                                                clean_wavs,
                                                                one_hot)
                    maeloss_val_sum += l1_loss_val
                    ce_val_sum += ce_loss_val

        del mix_wavs, clean_wavs, aux, aux_len, rec_app_wav, val

        loss_tr_list.append(mseloss_tr_sum / batch_num)
        loss_val_list.append(maeloss_val_sum / batch_num_val)

        ce_tr_list.append(ce_tr_sum / batch_num)
        ce_val_list.append(ce_val_sum / batch_num)
        epoch += 1

        if (maeloss_val_sum / batch_num_val) <= best_val_loss:
            best_val_loss = maeloss_val_sum / batch_num_val
            save_checkpoint('spexval', model, check_path, opt, no_improve_epochs, epoch, batch_si,
                            [loss_tr_list, loss_val_list], [ce_tr_list, ce_val_list],
                            [best_val_loss])
            print("\033[1;33m save best val\033[0m", str(best_val_loss))
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            print(f'Continue for {no_improve_epochs} epochs. ')
        if no_improve_epochs >= patience:
            print(f'Validation performance didn\'t improve for {patience} epochs. Stopping training.')
            break

        print('epoch:%d, va_mae_loss:%f, va_ce_loss:%f,' % (epoch, maeloss_val_sum / batch_num, ce_val_sum / batch_num))

        save_checkpoint('spexlast' + str(epoch), model, check_path, opt, no_improve_epochs, epoch, batch_si,
                        [loss_tr_list, loss_val_list],
                        [ce_tr_list, ce_val_list],
                        [best_val_loss])
        save_checkpoint('spexlast', model, check_path, opt, no_improve_epochs, epoch, batch_si,
                        [loss_tr_list, loss_val_list],
                        [ce_tr_list, ce_val_list],
                        [best_val_loss])


if __name__ == '__main__':
    run(0, lr=1e-4, e_num=50)
    run(lr=1e-4, e_num=50)
    run(lr=5e-5)
    run(lr=5e-5)
    run(lr=5e-5)