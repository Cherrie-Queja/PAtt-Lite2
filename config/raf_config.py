import argparse
import datetime

result_path = '/home/panr/all_results/PAtt-Lite'
now = datetime.datetime.now()
time_str = now.strftime("[%m-%d]-[%H-%M]_")
evaluate_path = None
# evaluate_path = result_path + '/checkpoint1/[06-02]-[22-10]-model_state_dict_best.pth'


parser = argparse.ArgumentParser()
parser.add_argument('--data_type', default='RAF-DB',
                    choices=['RAF-DB', 'AffectNet-7', 'CAER-S', 'FERPlus', 'FERPlusCNTK', 'FERPlus_wk', 'ExpW'],
                    type=str, help='dataset option')
parser.add_argument('--checkpoint_path', type=str, default=result_path + '/checkpoint/' + time_str + 'model.pth')
parser.add_argument('--best_checkpoint_path', type=str,
                    default=result_path + '/checkpoint/' + time_str + 'model_best.pth')
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N', help='number of data loading workers')
parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--img_size', default=112, type=int, metavar='N', help='image size')
parser.add_argument('--num-classes', default=7, type=int, metavar='N', help='number of expression classes')
parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N')  # original 144
parser.add_argument('--optimizer', type=str, default="adam", help='Optimizer, adam or sgd.')
parser.add_argument('--lr', '--learning-rate', default=3.5e-5, type=float, metavar='LR',
                    dest='lr')  # original:3.5e-5  AffectNet:1e-6
parser.add_argument('--momentum', default=0.9, type=float, metavar='M')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W', dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=30, type=int, metavar='N', help='print frequency')
parser.add_argument('--resume', default=None, type=str, metavar='PATH', help='path to checkpoint')
parser.add_argument('-e', '--evaluate', default=evaluate_path,
                    type=str, help='evaluate model on test set')
parser.add_argument('--beta', type=float, default=0.6)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--recon', type=bool, default=False, help='if recon features to classification or not')
args = parser.parse_args()

model_configs = f'{args.data_type}_ir50_focalloss_sam_b{args.batch_size}_lr{args.lr}'
