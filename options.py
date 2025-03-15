import argparse

class TrainOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        # data loader related
        self.parser.add_argument('--dataroot', type=str, default='', help='path of data')
        
        # train
        # self.parser.add_argument('--batch_size', type=int, default=4, help='batch size')
        # self.parser.add_argument('--lr', type=int, default=0.0001, help='Learning rate')
        # self.parser.add_argument('--total_epoch', type=int, default=50)
        # self.parser.add_argument('--gap_epoch', type=int, default=35)
        # self.parser.add_argument('--step_size', type=int, default=20)
        # self.parser.add_argument('--gamma', type=int, default=0.5)
        # self.parser.add_argument('--weight_decay', type=int, default=0)
        
        # test
        self.parser.add_argument('--ckpt_path', type=str, default="FDFuse_2024.pth", help='Path to the pre-trained weights')
        self.parser.add_argument('--vi_path', type=str, default="./vi", help='Path to the Visible images')
        self.parser.add_argument('--ir_path', type=str, default="./vi", help='Path to the Infrared images')
        self.parser.add_argument('--out_path', type=str, default="./out", help='Path to save the Fusion results')
        
        
    def parse(self):
        self.opt = self.parser.parse_args()
        args = vars(self.opt)
        print('\n--- load options ---')
        for name, value in sorted(args.items()):
          print('%s: %s' % (str(name), str(value)))
        return self.opt
