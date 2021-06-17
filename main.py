"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

from flask import Flask, jsonify, request

import os
import argparse
import io
from base64 import encodebytes
from PIL import Image

from munch import Munch
from torch.backends import cudnn
import torch

from datetime import datetime

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

        # STEP 1: 요청한 사진과 관련 정보 받고, assets/representative/resume/src 폴더에 사진 저장
        photo = request.files['photo']
        # id = request.form['id']
        # hairstyle = request.form['hairstyle']

        arguments.selected_hairstyle = 'mid' # TODO: 안드로이드에서 헤어스타일 도메인 받기, 현재는 임시
        save_image_path = './assets/representative/resume/src/' + arguments.selected_hairstyle + '/'
        date = str(datetime.today().year) + '{:02d}'.format(datetime.today().month) + '{:02d}'.format(datetime.today().day)
        image_name = date + '.jpg' # 사진이름 동적으로 생성 ex. 아이디+날짜시간 TODO: 아이디 추가하기
        url = save_image_path + image_name
        with open(url, 'wb') as f:
            f.write(photo.read())

        print("FINISH STEP1")

        # STEP 2: 사람이 감지되는지 확인
        number_of_face_detection = detect_faces(url)
        if number_of_face_detection >= 2: # 2인 이상 감지되었을 때
            print(number_of_face_detection)
            return jsonify({ 'error message': '2인 이상 감지되었습니다.' }), 400
        elif number_of_face_detection == 0: # 사람이 아무도 감지되지 않았을 때
            print(number_of_face_detection)
            return jsonify({ 'error message': '얼굴 인식에 실패하였습니다.' }), 400

        print("FINISH STEP2")

        # STEP 3: 전처리(얼굴 가운데로 맞추는) 실행 후 origin image에 덮어쓰기
        preprocessing_crop(url) # TODO: 도메인별(여-남, 헤어스타일 등)에 따라 다른 값 주기

        print("FINISH STEP3")
        
        # STEP 4: 모델을 통해 resume photo 생성
        # TODO: 사진 저장 시, 결과 사진 한 장만 저장
        arguments.result_image_name = image_name # 저장될 이미지 파일 이름 지정
        print("----- Start creating resume photo!! -----")
        main(arguments)
        print("----- Finish creating resume photo!! -----")

        print("FINISH STEP4")

        result_image = './expr/results/resume/'+image_name[:-4]+'3'+image_name[-4:] # TODO: 파일 이름 랜덤으로 secure하도록
        pil_img = Image.open(result_image, mode='r') # reads the PIL image
        byte_arr = io.BytesIO()
        pil_img.save(byte_arr, format='PNG') # convert the PIL image to byte array
        encoded_img = encodebytes(byte_arr.getvalue()).decode('ascii') # encode as base64

        # STEP 5: assets/representative/resume/src 폴더에 저장된 사진 삭제
        if os.path.exists(url):
            os.remove(url)
        else:
            print("The file does not exist")

        print("FINISH STEP6")

        return jsonify({ 'OK': '취업사진이 생성되었습니다.', 'photo': encoded_img }), 200

    elif request.method == 'GET':
        print("")

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
    parser.add_argument('--selected_hairstyle', type=str, default='short', help='Selected hairstyle by user')

    global arguments
    arguments = parser.parse_args()

    app.run(host="127.0.0.1", port="5000", debug=True)
