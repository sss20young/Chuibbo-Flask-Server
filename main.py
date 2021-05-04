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

from img_processing import detect_faces, preprocessing_crop


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

    if args.mode == 'sample':
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
    else:
        raise NotImplementedError
    return "A"


@app.route('/api/resume_photo/', methods=['GET', 'POST'])
def resume_photo():
    if request.method == 'POST':
        # TODO: STEP 1: 안드로이드에서 요청한 사진 받고, assets/representative/resume/src 폴더에 사진 저장
        # photo = flask.request.files.get('image')
        save_image_path = './assets/representative/resume/src/' + 'long/'
        image_name = 'e.jpg' # TODO: 사진이름 동적으로 생성 ex. 아이디+날짜시간
        url = save_image_path + image_name
        # photo.save(url)

        # encodedImg = request.form['file'] # 'file' is the name of the parameter you used to send the image
        # imgdata = base64.b64decode(encodedImg)
        # filename = 'image.jpg'  # choose a filename. You can send it via the request in an other variable
        # with open(filename, 'wb') as f:
        #     f.write(imgdata)

        # STEP 2: 사람이 둘 이상 감지되는지 확인
        number_of_face_detection = detect_faces(url)
        if number_of_face_detection >= 2:
            print(number_of_face_detection) # TODO: 2인 감지됐을 때, 어떻게 할 것인지
        elif number_of_face_detection == 1:
            print(number_of_face_detection)
        else:
            print(number_of_face_detection) # TODO: 0인 감지됐을 때, 어떻게 할 것인지


        # STEP 3: 전처리(얼굴 가운데로 맞추는) 실행 후 origin image에 덮어쓰기
        preprocessing_crop(url) # TODO: 도메인별(여-남, 헤어스타일 등)에 따라 다른 값 주기
        
        # STEP 4: 모델을 통해 resume photo 생성
        # TODO: 사진 저장 시, 결과 사진 한 장만 저장
        # TODO: assets/representative/resume/ref 폴더에 합성할 사진 넣어주고 그 중 선택
        arguments.result_image_name = image_name # 저장될 이미지 파일 이름 지정
        print("----- Start creating resume photo!! -----")
        main(arguments)
        print("----- Finish creating resume photo!! -----")

        return "<h1>Success</h1>"

    elif request.method == 'GET':
        # TODO: STEP 5: 생성된 사진 전송
        print("----- Start sending resume photo!! -----")

        # TODO: STEP 6: assets/representative/resume/src 폴더에 저장된 사진 삭제
        

if __name__ == '__main__':
    print("* Loading GAN model and Flask starting server... please wait until server has fully started")

    parser = argparse.ArgumentParser()

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
    parser.add_argument('--sample_dir', type=str, default='expr/samples', help='Directory for saving generated images')
    parser.add_argument('--checkpoint_dir', type=str, default='expr/checkpoints/resume', help='Directory for saving network checkpoints')

    # directory for testing
    parser.add_argument('--result_dir', type=str, default='expr/results/resume', help='Directory for saving generated images and videos')
    parser.add_argument('--src_dir', type=str, default='assets/representative/resume/src', help='Directory containing input source images')
    parser.add_argument('--ref_dir', type=str, default='assets/representative/resume/ref', help='Directory containing input reference images')

    # face alignment
    parser.add_argument('--wing_path', type=str, default='expr/checkpoints/wing.ckpt')
    parser.add_argument('--lm_path', type=str, default='expr/checkpoints/celeba_lm_mean.npz')

    # 저장될 이미지 파일 이름
    parser.add_argument('--result_image_name', type=str, default='.jpg', help='File for resulting image')

    global arguments
    arguments = parser.parse_args()

    app.run(host="127.0.0.1", port="5000", debug=True)
