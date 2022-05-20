import argparse
import matplotlib.pyplot as plt
import matplotlib
import os
import json
import numpy as np


def get_args_parser():
    parser = argparse.ArgumentParser('', add_help=False)
    parser.add_argument('--experiment_names', type=str, nargs='*', help='')
    parser.add_argument('--alias_names', type=str, nargs='*', help='')
    parser.add_argument('--output_path', type=str, help='')
    parser.add_argument('--input_path', type=str, help='')
    parser.add_argument('--name', default="", type=str, help='')
    return parser


def get_loss_data(data, batch_size = 1):
    tr_loss = []
    ts_loss = []
    epoch = []
    lr = []
    for d in data:
        if 'train_loss' in d:
            tr_loss.append(d['train_loss']/batch_size)
        if 'test_loss' in d:
            ts_loss.append(d['test_loss'])
        if 'epoch' in d:
            epoch.append(d['epoch'])
        if 'train_lr' in d:
            lr.append(d['train_lr'])

    return tr_loss, ts_loss, epoch, lr


def main(args):
    experiment_names = args.experiment_names
    alias_names = args.alias_names
    output_path = args.output_path
    file_name = ''
    file_name_predictions = ''
    data_paths = args.input_path

    loss_file_name = 'log.txt'
    input_params_file_name = 'input_log.txt'
    name = args.name
    #name = '-' + name[1::]
    pred_file_name = '_prediction_out' + name + '.txt'
    input_params = ['lr', 'lr_backbone', 'batch_size', 'epochs', 'lr_drop', 'batch_size', 'transformations', 'augmentations', 'sgd', 'scheduler', 'enc_layers', 'dec_layers', 'nheads', 'time', 'backbone', 'crop_w']

    if alias_names:
        name_to_alias = dict(zip(experiment_names, alias_names))
        file_name = '-'.join(alias_names) + '.png'
        file_name_predictions = '-'.join(alias_names) + '.txt'
    else:
        name_to_alias = dict(zip(experiment_names, experiment_names))
        file_name = '-'.join(experiment_names) + '.png'
        file_name_predictions = '-'.join(alias_names) + '.txt'
    loss_paths = []
    input_param_path = []
    for en in experiment_names:
        loss_paths.append(os.path.join(data_paths, en, loss_file_name))
        input_param_path.append(os.path.join(data_paths, en, input_params_file_name))


    loss_files = []
    input_param_files = []
    for i in range(len(loss_paths)):
        lp = loss_paths[i]
        ip = input_param_path[i]

        with open(lp, 'r') as f:
            d = f.read().split('\n')
            d_lst = []
            for di in d:
                if di.strip():
                    d_lst.append(json.loads(di.strip()))
            loss_files.append(d_lst)

        with open(ip, 'r') as f:
            input_param_files.append(json.load(f))
                                        # 12, 25
    fig, axs = plt.subplots(4,1, figsize=(12,36),  gridspec_kw={'height_ratios': [3, 3, 3, 3]})

    plt.rcParams.update({'font.size': 14})

    if alias_names:
        col_names = alias_names
    else:
        col_names = experiment_names
    row_names = input_params
    val3 = [["" for c in range(len(col_names))] for r in range(len(row_names))]



    for i in range(len(experiment_names)):
        inp_data = input_param_files[i]
        tr_loss, ts_loss, epoch, lr = get_loss_data(loss_files[i], batch_size=inp_data['batch_size'])
        axs[1].plot(epoch, tr_loss, label=name_to_alias[experiment_names[i]])
        axs[2].plot(epoch, ts_loss, label=name_to_alias[experiment_names[i]])
        axs[3].plot(epoch, lr, label=name_to_alias[experiment_names[i]])
        for j, ip in enumerate(input_params):
            if ip in inp_data:
                if ip == 'scheduler' and len(inp_data[ip]) > 1:
                    val3[j][i] = str(inp_data[ip][0])

                else:
                    val3[j][i] = str(inp_data[ip])

                if ip == 'crop_w':
                    val3[j][i] = '[' + str(inp_data[ip]) + ', ' + str(inp_data['crop_h']) + ']'
                    

    axs[1].legend()
    axs[1].grid()
    axs[1].set(xlabel ='Epoch', ylabel ='Loss',
               title ='Train set loss')
    axs[2].legend()
    axs[2].grid()
    axs[2].set(xlabel ='Epoch', ylabel ='Loss',
               title ='Validation set loss')
    axs[3].legend()
    axs[3].grid()
    axs[3].set(xlabel ='Epoch', ylabel ='Learning Rate',
               title ='Learning Rate')



    axs[0].set_axis_off()
    table = axs[0].table(
        cellText = val3,
        rowLabels = row_names,
        colLabels = col_names,
        rowColours =["palegreen"] * len(row_names),
        colColours =["palegreen"] * len(col_names),
        cellLoc ='center',
        loc ='upper left')
    table.scale(len(experiment_names) * 0.25, 2)
    #axs[0].set_title('matplotlib.axes.Axes.table() function Example', fontweight ="bold")

    fig.savefig(os.path.join(output_path, file_name), dpi=300, bbox_inches='tight')



    
    loss_paths = []
    validation_results = []

    for en in experiment_names:
        loss_paths.append(os.path.join(data_paths, en, pred_file_name))

    for p in loss_paths:
        with open(p) as f:
            res = []
            for l in f:
                if 'Average' in l:
                    ll = l.strip()
                    lll = ll.split('=')[4].strip()
                    res.append(float(lll))
            validation_results.append(res)

    full_string = ''
    names_len = [] 
    for i in range(len(validation_results)):
        exp_name = experiment_names[i]
        le = len(exp_name)
        names_len.append(le)
        vr = validation_results[i]
        st = f'''
            | {' '*le} | AP    | AP .5 | AP .75 | AP (M) | AP (L) | AR    | AR .5 | AR .75 | AR (M) | AR (L) |
            |-{'-'*le}-|-------|-------|--------|--------|--------|-------|-------|--------|--------|--------|
            | {exp_name} | {vr[0]:.3f} | {vr[1]:.3f} | {vr[2]:.3f}  | {vr[3]:.3f}  | {vr[4]:.3f}  | {vr[5]:.3f} | {vr[6]:.3f} | {vr[7]:.3f}  | {vr[8]:.3f}  | {vr[9]:.3f}  |
            \n'''
        full_string += st

    max_len = np.max(names_len)
    full_string += f'''\n\n
            | {' '*max_len} | AP    | AP .5 | AP .75 | AP (M) | AP (L) | AR    | AR .5 | AR .75 | AR (M) | AR (L) |'''
    for i in range(len(validation_results)):
        exp_name = experiment_names[i]
        le = len(exp_name)
        vr = validation_results[i]
        st = f'''
            |-{'-'*max_len}-|-------|-------|--------|--------|--------|-------|-------|--------|--------|--------|
            | {exp_name}{' '*(max_len-le)} | {vr[0]:.3f} | {vr[1]:.3f} | {vr[2]:.3f}  | {vr[3]:.3f}  | {vr[4]:.3f}  | {vr[5]:.3f} | {vr[6]:.3f} | {vr[7]:.3f}  | {vr[8]:.3f}  | {vr[9]:.3f}  |'''
        full_string += st

    with open(os.path.join(output_path, args.name + file_name_predictions), 'w') as f:
        f.write(full_string)
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser('', parents=[get_args_parser()])
    args = parser.parse_args()

    main(args)



