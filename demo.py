import argparse

import cv2
import numpy as np
import pyrealsense2 as rs
import torch
from openvino.inference_engine import IECore

from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.keypoints import extract_keypoints, group_keypoints
from modules.load_state import load_state
from modules.pose import Pose, track_poses
from val import normalize, pad_width


class ImageReader(object):
    def __init__(self, file_names):
        self.file_names = file_names
        self.max_idx = len(file_names)

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx == self.max_idx:
            raise StopIteration
        img = cv2.imread(self.file_names[self.idx], cv2.IMREAD_COLOR)
        if img.size == 0:
            raise IOError('Image {} cannot be read'.format(self.file_names[self.idx]))
        self.idx = self.idx + 1
        return img


class VideoReader(object):
    def __init__(self, file_name):
        self.file_name = file_name
        try:  # OpenCV needs int to read from webcam
            self.file_name = int(file_name)
        except ValueError:
            pass

    def __iter__(self):
        self.cap = cv2.VideoCapture(self.file_name)
        if not self.cap.isOpened():
            raise IOError('Video {} cannot be opened'.format(self.file_name))
        return self

    def __next__(self):
        was_read, img = self.cap.read()
        if not was_read:
            raise StopIteration
        return img


def infer_fast(net, img, net_input_height_size, stride, upsample_ratio, cpu,
               pad_value=(0, 0, 0), img_mean=(128, 128, 128), img_scale=1/256, openvino=False):
    height, width, _ = img.shape
    scale = net_input_height_size / height

    scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    if not openvino:  # OpenVINO model doesn't require input to be normalized
        scaled_img = normalize(scaled_img, img_mean, img_scale)
    min_dims = [net_input_height_size, max(scaled_img.shape[1], net_input_height_size)]
    padded_img, pad = pad_width(scaled_img, stride, pad_value, min_dims)

    if openvino:
        reshaped_img = np.transpose(padded_img, (2, 0, 1))
        res = net.infer(inputs={next(iter(net.inputs)): reshaped_img})
        pafs = res['Mconv7_stage2_L1']
        heatmaps = res['Mconv7_stage2_L2']
        pafs = np.transpose(pafs.squeeze(), (1, 2, 0))
        heatmaps = np.transpose(heatmaps.squeeze(), (1, 2, 0))
    else:
        tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()
        if not cpu:
            tensor_img = tensor_img.cuda()

        stages_output = net(tensor_img)

        stage2_heatmaps = stages_output[-2]
        heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))

        stage2_pafs = stages_output[-1]
        pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))

    heatmaps = cv2.resize(heatmaps, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)
    pafs = cv2.resize(pafs, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    return heatmaps, pafs, scale, pad


def init_realsense():
    image_width = 640
    image_height = 480
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, image_width, image_height, rs.format.bgr8, 30)
    _ = pipeline.start(config)
    return pipeline


def run_demo(net, height_size, cpu, track, smooth, image_provider=None, use_realsense_cam=False, openvino=False):
    if not image_provider and not use_realsense_cam:
        raise ValueError('Either `image_provider` or `use_realsense_cam` must be provided')

    if use_realsense_cam:
        pipeline = init_realsense()

    stride = 8
    upsample_ratio = 4
    num_keypoints = Pose.num_kpts
    previous_poses = []
    delay = 33
    while True:

        # get frame from source provided
        if use_realsense_cam:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            img = np.asanyarray(color_frame.get_data())
        else:
            img = next(image_provider)

        orig_img = img.copy()
        heatmaps, pafs, scale, pad = infer_fast(net, img, height_size, stride, upsample_ratio, cpu, openvino=openvino)

        total_keypoints_num = 0
        all_keypoints_by_type = []
        for kpt_idx in range(num_keypoints):  # 19th for bg
            total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)

        pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs, demo=True)
        for kpt_id in range(all_keypoints.shape[0]):
            all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * stride / upsample_ratio - pad[1]) / scale
            all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * stride / upsample_ratio - pad[0]) / scale
        current_poses = []
        for n in range(len(pose_entries)):
            if len(pose_entries[n]) == 0:
                continue
            pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
            for kpt_id in range(num_keypoints):
                if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                    pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                    pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
            pose = Pose(pose_keypoints, pose_entries[n][18])
            current_poses.append(pose)

        if track:
            track_poses(previous_poses, current_poses, smooth=smooth)
            previous_poses = current_poses
        for pose in current_poses:
            pose.draw(img)
        img = cv2.addWeighted(orig_img, 0.6, img, 0.4, 0)
        for pose in current_poses:
            cv2.rectangle(img, (pose.bbox[0], pose.bbox[1]),
                          (pose.bbox[0] + pose.bbox[2], pose.bbox[1] + pose.bbox[3]), (0, 255, 0))
            if track:
                cv2.putText(img, 'id: {}'.format(pose.id), (pose.bbox[0], pose.bbox[1] - 16),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))
        cv2.imshow('Lightweight Human Pose Estimation Python Demo', img)
        key = cv2.waitKey(delay)
        if key == 27:  # esc
            return
        elif key == 112:  # 'p'
            if delay == 33:
                delay = 0
            else:
                delay = 33


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''Lightweight human pose estimation python demo.
                       This is just for quick results preview.
                       Please, consider c++ demo for the best performance.''')
    parser.add_argument('--checkpoint-path', type=str, help='Required if using pytorch for inference. Path to the pytorch checkpoint.')
    parser.add_argument('--height-size', type=int, default=256, help='network input layer height size')
    parser.add_argument('--video', type=str, default='', help='path to video file or camera id')
    parser.add_argument('--images', nargs='+', default='', help='path to input image(s)')
    parser.add_argument('--realsense-cam', action='store_true', help='use an Intel Realsense camera as live video source')
    parser.add_argument('--cpu', action='store_true', help='run network inference on cpu')
    parser.add_argument('--track', type=int, default=1, help='track pose id in video')
    parser.add_argument('--smooth', type=int, default=1, help='smooth pose keypoints')
    parser.add_argument('--openvino', action='store_true', help='use OpenVINO for model inference instead of pytorch')
    parser.add_argument('--xml-path', type=str, help='Required if --openvino flag is given. Path to the Human Pose Estimation model (.xml) file.')
    parser.add_argument('--bin-path', type=str, help='Required if --openvino flag is given. Path to the Human Pose Estimation weights (.bin) file.')
    args = parser.parse_args()

    if args.video == '' and args.images == '' and not args.realsense_cam:
        raise ValueError('One of [--video, --image, --realsense-cam] must be provided')

    if args.openvino:
        if args.xml_path is None or args.bin_path is None:
            raise ValueError('--xml-path and --bin-path must be provided if using OpenVINO for inference')
        ie = IECore()
        ie_net = ie.read_network(model=args.xml_path, weights=args.bin_path)
        input_layer_name = next(iter(ie_net.inputs))
        ie_net.reshape({input_layer_name: (1, 3, 256, 344)})
        net = ie.load_network(network=ie_net, device_name='CPU')
    else:
        if args.checkpoint_path == '':
            raise ValueError('--checkpoint-path must be provided if using pytorch for inference')
        net = PoseEstimationWithMobileNet()
        checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
        load_state(net, checkpoint)
        net = net.eval()
        if not args.cpu:
            net = net.cuda()

    if args.realsense_cam:
        run_demo(net, args.height_size, args.cpu, args.track, args.smooth, use_realsense_cam=True, openvino=args.openvino)
    else:
        frame_provider = ImageReader(args.images)
        if args.video != '':
            frame_provider = VideoReader(args.video)
        else:
            args.track = 0
        run_demo(net, args.height_size, args.cpu, args.track, args.smooth, image_provider=frame_provider, openvino=args.openvino)
