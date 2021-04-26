"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

from flask import Flask
from flask import request

import os
import argparse

from munch import Munch
from torch.backends import cudnn
import torch

from core.data_loader import get_train_loader
from core.data_loader import get_test_loader
from core.solver import Solver

from preprocessing import preprocessing_crop


app = Flask(__name__)

def str2bool(v):
    return v.lower() in ('true')

def subdirs(dname):
    return [d for d in os.listdir(dname)
            if os.path.isdir(os.path.join(dname, d))]

def main(args):
    cudnn.benchmark = True
    torch.manual_seed(args.seed)

    solver = Solver(args)

    if args.mode == 'train':
        assert len(subdirs(args.train_img_dir)) == args.num_domains
        assert len(subdirs(args.val_img_dir)) == args.num_domains
        loaders = Munch(src=get_train_loader(root=args.train_img_dir,
                                             which='source',
                                             img_size=args.img_size,
                                             batch_size=args.batch_size,
                                             prob=args.randcrop_prob,
                                             num_workers=args.num_workers),
                        ref=get_train_loader(root=args.train_img_dir,
                                             which='reference',
                                             img_size=args.img_size,
                                             batch_size=args.batch_size,
                                             prob=args.randcrop_prob,
                                             num_workers=args.num_workers),
                        val=get_test_loader(root=args.val_img_dir,
                                            img_size=args.img_size,
                                            batch_size=args.val_batch_size,
                                            shuffle=True,
                                            num_workers=args.num_workers))
        solver.train(loaders)
    elif args.mode == 'sample':
        assert len(subdirs(args.src_dir)) == args.num_domains
        assert len(subdirs(args.ref_dir)) == args.num_domains
        loaders = Munch(src=get_test_loader(root=args.src_dir,
                                            img_size=args.img_size,
                                            batch_size=args.val_batch_size,
                                            shuffle=False,
                                            num_workers=args.num_workers),
                        ref=get_test_loader(root=args.ref_dir,
                                            img_size=args.img_size,
                                            batch_size=args.val_batch_size,
                                            shuffle=False,
                                            num_workers=args.num_workers))
        solver.sample(loaders)
    elif args.mode == 'eval':
        solver.evaluate()
    elif args.mode == 'align':
        from core.wing import align_faces
        align_faces(args, args.inp_dir, args.out_dir)
    else:
        raise NotImplementedError
    return "A"

# @app.route('/api/resume_photo/', method=['GET', 'POST'])
@app.route('/api/resume_photo/')
def resume_photo():
    if request.method == 'POST':
        # TODO: STEP 1: 사진 받고, assets/representative/resume/src 폴더에 사진 저장
        # photo = flask.request.files.get('image')
        # photo.save('./assets/representative/resume/src')

        # encodedImg = request.form['file'] # 'file' is the name of the parameter you used to send the image
        # imgdata = base64.b64decode(encodedImg)
        # filename = 'image.jpg'  # choose a filename. You can send it via the request in an other variable
        # with open(filename, 'wb') as f:
        #     f.write(imgdata)

        # TODO: STEP 2: 사람이 둘 이상 감지되는지 확인

        # TODO: STEP 3: 사람얼굴인지 인식하고, 사람얼굴이면 전처리 실행 후 덮어쓰기
        # preprocessing_crop() # TODO: 도메인별(여-남, 헤어스타일 등)에 따라 다른 값 주기
        
        # STEP 4: 모델을 통해 resume photo 생성
        # TODO: assets/representative/resume/ref 폴더에 합성할 사진 넣어주고 그 중 선택
        print("----- Start creating resume photo!! -----")
        main(arguments)

        print("----- Finish creating resume photo!! -----")

    elif request.method == 'GET':
        # TODO: STEP 5: 생성된 사진 전송
        print("----- Start sending resume photo!! -----")
        main(arguments)

        # TODO: STEP 6: assets/representative/resume/src 폴더에 저장된 사진 삭제
        

if __name__ == '__main__':
    print("* Loading GAN model and Flask starting server... please wait until server has fully started")

    parser = argparse.ArgumentParser()

    # TODO: 필요한 argument 속성만 남겨두기
    # model arguments
    parser.add_argument('--img_size', type=int, default=256, help='Image resolution')
    parser.add_argument('--num_domains', type=int, default=3, help='Number of domains')
    parser.add_argument('--latent_dim', type=int, default=16, help='Latent vector dimension')
    parser.add_argument('--hidden_dim', type=int, default=512, help='Hidden dimension of mapping network')
    parser.add_argument('--style_dim', type=int, default=64, help='Style code dimension')

    # weight for objective functions
    parser.add_argument('--lambda_reg', type=float, default=1, help='Weight for R1 regularization')
    parser.add_argument('--lambda_cyc', type=float, default=1, help='Weight for cyclic consistency loss')
    parser.add_argument('--lambda_sty', type=float, default=1, help='Weight for style reconstruction loss')
    parser.add_argument('--lambda_ds', type=float, default=1, help='Weight for diversity sensitive loss')
    parser.add_argument('--ds_iter', type=int, default=100000, help='Number of iterations to optimize diversity sensitive loss')
    parser.add_argument('--w_hpf', type=float, default=1, help='weight for high-pass filtering')

    # training arguments
    parser.add_argument('--randcrop_prob', type=float, default=0.5, help='Probabilty of using random-resized cropping')
    parser.add_argument('--total_iters', type=int, default=100000, help='Number of total iterations')
    parser.add_argument('--resume_iter', type=int, default=0, help='Iterations to resume training/testing')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--val_batch_size', type=int, default=32, help='Batch size for validation')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for D, E and G')
    parser.add_argument('--f_lr', type=float, default=1e-6, help='Learning rate for F')
    parser.add_argument('--beta1', type=float, default=0.0, help='Decay rate for 1st moment of Adam')
    parser.add_argument('--beta2', type=float, default=0.99, help='Decay rate for 2nd moment of Adam')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for optimizer')
    parser.add_argument('--num_outs_per_domain', type=int, default=10, help='Number of generated images per domain during sampling')

    # misc
    # parser.add_argument('--mode', type=str, required=True, choices=['train', 'sample', 'eval', 'align'], help='This argument is used in solver')
    parser.add_argument('--mode', type=str, default='sample', help='This argument is used in solver')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers used in DataLoader')
    parser.add_argument('--seed', type=int, default=777, help='Seed for random number generator')

    # directory for training
    parser.add_argument('--train_img_dir', type=str, default='data/celeba_hq/train', help='Directory containing training images')
    parser.add_argument('--val_img_dir', type=str, default='data/celeba_hq/val', help='Directory containing validation images')
    parser.add_argument('--sample_dir', type=str, default='expr/samples', help='Directory for saving generated images')
    parser.add_argument('--checkpoint_dir', type=str, default='expr/checkpoints/resume', help='Directory for saving network checkpoints')

    # directory for calculating metrics
    parser.add_argument('--eval_dir', type=str, default='expr/eval', help='Directory for saving metrics, i.e., FID and LPIPS')

    # directory for testing
    parser.add_argument('--result_dir', type=str, default='expr/results/resume', help='Directory for saving generated images and videos')
    parser.add_argument('--src_dir', type=str, default='assets/representative/resume/src', help='Directory containing input source images')
    parser.add_argument('--ref_dir', type=str, default='assets/representative/resume/ref', help='Directory containing input reference images')
    #parser.add_argument('--inp_dir', type=str, default='assets/representative/custom/female', help='input directory when aligning faces')
    #parser.add_argument('--out_dir', type=str, default='assets/representative/resume/src/female', help='output directory when aligning faces')

    # face alignment
    parser.add_argument('--wing_path', type=str, default='expr/checkpoints/wing.ckpt')
    parser.add_argument('--lm_path', type=str, default='expr/checkpoints/celeba_lm_mean.npz')

    # step size
    parser.add_argument('--print_every', type=int, default=10)
    parser.add_argument('--sample_every', type=int, default=5000)
    parser.add_argument('--save_every', type=int, default=10000)
    parser.add_argument('--eval_every', type=int, default=50000)

    global arguments
    arguments = parser.parse_args()

    app.run(host="127.0.0.1", port="5000", debug=True)
