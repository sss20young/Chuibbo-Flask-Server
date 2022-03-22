"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.
This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

from flask import Flask, jsonify, request, Response
from flask_restx import Api, Resource

import os
import argparse
import numpy as np
import json
import cv2 as cv
from lip import makeUp

from munch import Munch
from torch.backends import cudnn
import torch

from datetime import datetime

from core.data_loader import get_test_loader
from core.solver import Solver
from img_processing import delete_image, detect_faces, preprocessing_crop, transform_encoded_image

from PIL import Image

import traceback, random
import math
import time

app = Flask(__name__)

""" swagger-ui """
api = Api(app, version='0.0.1', title='[ Chuibbo-Flask ]', description='Chuibbo API 명세서', doc="/swagger-ui")
resume_photo_api = api.namespace('api/resume-photos', description='Resume Photo Controller')
upload_api = api.namespace('api/upload', description='Upload Controller')
parameter_api = api.namespace('api/parameter', description='Parameter Controller')
strong_api = api.namespace('api/strong', description='Strong Controller')
makeup_api = api.namespace('api/makeup', description='Makeup Controller')

""" MAKE UP """
def data(size, r, g, b, index):
    global GS, GR, GG, GB, ID
    GS = size
    GR = r
    GG = g
    GB = b
    ID = index

def dataStrong(strong):
    global GT
    GT = strong

""" RESUME PHOTO """
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

hair_dict = {'long' : 1, 'mid' : 2, 'short' : 3}

""" RESUME PHOTO API """
@resume_photo_api.route('')
class ResumePhoto(Resource):
    def post(self):

        start = time.time()
        math.factorial(100000)
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
        arguments, unknown = parser.parse_known_args()

        # STEP 1: 요청한 사진과 관련 정보 받고, assets/representative/resume/src 폴더에 사진 저장
        photo = request.files['photo']
        id = request.form['id']
        sex = request.form['sex']
        face_shape = request.form['face_shape']
        hairstyle = request.form['hairstyle'] # 취업 사진 헤어스타일
        prev_hairstyle = request.form['prev_hairstyle']
        # prev_hairstyle = request.form['prev_hairstyle'] # TODO: 기존의 본인 헤어스타일
        suit = request.form['suit']
        src_tail_num = str(random.randrange(1, 250))
        arguments.src_dir = 'assets/representative/resume/src' + src_tail_num
        os.makedirs(arguments.src_dir, exist_ok=True)
        os.makedirs(arguments.src_dir+"/long", exist_ok=True)
        os.makedirs(arguments.src_dir+"/mid", exist_ok=True)
        os.makedirs(arguments.src_dir+"/short", exist_ok=True)

        arguments.selected_hairstyle = hairstyle
        src_dir_path = f'./assets/representative/resume/src{src_tail_num}/{prev_hairstyle}/'
        date = '{:04d}'.format(datetime.today().year) + '{:02d}'.format(datetime.today().month) + '{:02d}'.format(datetime.today().day) + '{:02d}'.format(datetime.today().hour) + '{:02d}'.format(datetime.today().minute) + str(random.randrange(1, 250))
        src_image = f'{id}_{date}.jpg' # 사진이름 동적으로 생성 ex. 아이디+날짜시간
        src_image_path = src_dir_path + src_image
        with open(src_image_path, 'wb') as f:
            f.write(photo.read())

        # 선언을 위한 할당(초기화)
        result_image_path = './expr/results/resume/example.jpg'
        result_image_png = './expr/results/resume/example.png'

        print("FINISH STEP1")

        try:
            # STEP 2: 사람이 감지되는지 확인
            number_of_face_detection = detect_faces(src_image_path)
            if number_of_face_detection >= 2: # 2인 이상 감지되었을 때
                print(number_of_face_detection)
                return jsonify({ 'code': 2, 'message': '2인 이상 감지되었습니다.'}), 200
            elif number_of_face_detection == 0: # 사람이 아무도 감지되지 않았을 때
                print(number_of_face_detection)
                return jsonify({ 'code': 3, 'message': '얼굴 인식에 실패하였습니다.'}), 200

            print("FINISH STEP2")

            # STEP 3: 전처리(얼굴 가운데로 맞추는) 실행 후 origin image에 덮어쓰기
            preprocessing_crop(src_image_path) # TODO: 도메인별(여-남, 헤어스타일 등)에 따라 다른 값 주기

            print("FINISH STEP3")

            # STEP 4: 모델을 통해 resume photo 생성
            # TODO: 사진 저장 시, 결과 사진 한 장만 저장
            arguments.result_image_name = src_image
            print("----- Start creating resume photo!! -----")
            main(arguments)

            print("FINISH STEP4")

            image_title, image_ext = os.path.splitext(src_image)
            # '_' + str(hair_dict[hairstyle]) +
            result_dir_path = './expr/results/resume'
            result_image_path = f'{result_dir_path}/{src_image}' # TODO: 파일 이름 랜덤으로 secure하도록
            result_image_jpg = Image.open(result_image_path)
            result_image_png = f'{result_dir_path}/{image_title}.png'
            result_image_jpg.save(result_image_png) # png로 변환
            encoded_img = transform_encoded_image(result_image_png)

        except:
            delete_image(src_image_path)
            os.rmdir(arguments.src_dir+"/long")
            os.rmdir(arguments.src_dir+"/mid")
            os.rmdir(arguments.src_dir+"/short")
            os.rmdir(arguments.src_dir)
            delete_image(result_image_path)
            delete_image(result_image_png)

            print(traceback.format_exc())

            return jsonify({ 'code': 4, 'message': '사진 합성 중 예기치 못한 오류가 발생하였습니다.'}), 200

        else:
            # STEP 5: assets/representative/resume/src 폴더에 저장된 사진 삭제
            delete_image(src_image_path)
            os.rmdir(arguments.src_dir+"/long")
            os.rmdir(arguments.src_dir+"/mid")
            os.rmdir(arguments.src_dir+"/short")
            os.rmdir(arguments.src_dir)
            delete_image(result_image_path)
            delete_image(result_image_png)

            print("FINISH STEP5")
            end = time.time()
            print(f"{end - start:.5f} sec")
            return json.dumps({ 'code': 1, 'message': '', 'data': encoded_img.decode('ascii') }), 200


""" MAKEUP API """
@upload_api.route('')
class Upload(Resource):
    def post(self):
        """메이크업할 사진 업로드"""
        # 이미지 받아오기
        img = cv.imdecode(np.fromstring(request.files['image'].read(), np.uint8), cv.IMREAD_UNCHANGED)
        w, h = img.shape[1], img.shape[0]
        if w-h > 1000:
            img = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)
        # 서버 내 이미지 저장
        path_file = ('static/Input.jpg')
        cv.imwrite(path_file, img)
        img_processed = json.dumps(path_file)
        # json string으로 돌려받기
        return Response(response=img_processed, status=200, mimetype="application/json")

@parameter_api.route('')
class Parameter(Resource):
    def post():
        """메이크업 RGB 색상값 전달"""
        #print(request.is_json)
        params = request.get_json()
        print(params)
        size = params.get('size')
        r = params.get('rColor')
        g = params.get('gColor')
        b = params.get('bColor')
        index = params.get('index')
        data(size, r, g, b, index)
        return Response()

@strong_api.route('')
class Strong(Resource):
    def post():
        """메이크업 강도 전달"""
        #print(request.is_json)
        params = request.get_json()
        #print(params/200)
        dataStrong(params/200)
        return Response()

@makeup_api.route('')
class Makeup(Resource):
    def post():
        """메이크업 실행"""
        params = request.get_json()
        print("메이크업 시작......")
        print(GR, GG, GB, GS, ID, GT)
        make = makeUp()
        make.readImg()  # 이미지 초기화
        make.makeUpFeatures(r=GR, g=GG, b=GB, size=(GS, GS), index=ID, strong=GT)
        return Response(response=params)


if __name__ == '__main__':
    app.run(host="127.0.0.1", port="5000", debug=True)
