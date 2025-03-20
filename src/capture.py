from datetime import datetime, timedelta
import cv2 as cv
from typing import List
import os
import numpy as np
import pyrealsense2 as rs
import time
import numpy as np
import argparse

class Camera():
    def __init__(
            self,
            id,
            name:str,
            max_w_h:int
        ):
        self.id = id
        self.name = name
        self.max_w_h = max_w_h

        self.record = False
        self.frames = []

    def captureNext(self):
        raise NotImplementedError()
    
    def save_frames(self):
        raise NotImplementedError()
    
    def __resize__(self, frame):
        f_h, f_w = frame.shape[0], frame.shape[1]
        if self.max_w_h is None:
            return frame
        scale = np.min([self.max_w_h / f_w, self.max_w_h / f_h])

        new_h, new_w = int(np.round(scale * f_h)), int(np.round(scale * f_w))
        frame = cv.resize(frame, (new_w, new_h))
        return frame

class RealSenseCamera(Camera):
    def __init__(
            self,
            serial_number:str,
            name:str,
            max_w_h:int,
            dataset_directory:str
        ):
        super(RealSenseCamera, self).__init__(serial_number, name, max_w_h)
        self.__pipeline__ = rs.pipeline()
        self.__config__ = rs.config()
        self.serialnumber = serial_number
        self.__config__.enable_device(serial_number)
        self.__config__.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
        self.__config__.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.__pipeline_profile__ = self.__pipeline__.start(self.__config__)

        # filter
        self.align = rs.align(rs.stream.color)  

        self.filter = []
        #decimation_filter = rs.decimation_filter()
        #decimation_filter.set_option(rs.option.filter_magnitude, 2.0)
        #self.filter.append(decimation_filter)

        threshold_filter = rs.threshold_filter()
        threshold_filter.set_option(rs.option.min_distance, 0.)#0.3)
        threshold_filter.set_option(rs.option.max_distance, 16.)#3.0)
        self.filter.append(threshold_filter)

        #disparity_transform = rs.disparity_transform()
        #self.filter.append(disparity_transform)

        # spatial_filter = rs.spatial_filter()
        # spatial_filter.set_option(rs.option.filter_magnitude, 2.0)
        # spatial_filter.set_option(rs.option.filter_smooth_alpha, 0.25)
        # spatial_filter.set_option(rs.option.filter_smooth_delta, 18.0)
        # self.filter.append(spatial_filter)

        # temporal_filter = rs.temporal_filter()
        # temporal_filter.set_option(rs.option.filter_smooth_alpha, 0.36)
        # temporal_filter.set_option(rs.option.filter_smooth_delta, 100.0)
        # self.filter.append(temporal_filter)
        self.dataset_directory = '' if dataset_directory is None else dataset_directory + '/'
        self.idx = 0
        for fname in os.listdir(f'{self.dataset_directory}sequences/realsense/new'):
            camera = fname[0:4]
            if camera != self.name:
                continue
            frame_idx = int(fname[5:10])
            if frame_idx > self.idx:
                self.idx = frame_idx + 1
        pass

    def __del__(self):
        self.__pipeline__.stop()

    def __apply_filter(self, depth_frame):
        for filter in self.filter:
            depth_frame = filter.process(depth_frame)
        return depth_frame
    
    def save_frames(self):        
        for color_frame, depth_frame in self.frames:
            fname_color = f'{self.dataset_directory}/sequences/realsense/new/{self.name}F{self.idx:0>5}_color.jpg'
            fname_depth = f'{self.dataset_directory}/sequences/realsense/new/{self.name}F{self.idx:0>5}_depth.jpg'

            color_frame = np.uint8(np.round(color_frame * 255))
            depth_frame = np.uint8(np.round(depth_frame * 255))

            color_frame = cv.cvtColor(color_frame, cv.COLOR_BGR2RGB)

            cv.imwrite(fname_color, color_frame)
            cv.imwrite(fname_depth, depth_frame)
            self.idx += 1
        self.frames = []

    def captureNext(self) -> tuple[np.array, np.array]:
        try:
            frames = self.__pipeline__.wait_for_frames()
            aligned_frames = self.align.process(frames)

            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            #depth_frame = self.__apply_filter(depth_frame)
            
            if depth_frame and color_frame:
                # Convert images to numpy arrays
                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())

                color_image = self.__resize__(color_image)
                depth_image = self.__resize__(depth_image)

                color_image = color_image / 255
                depth_image = depth_image #/ np.iinfo(np.uint16).max

                if self.record:
                    self.frames.append((color_image, depth_image))

                #if len(self.frames) > 500:
                #    self.save_frames()

                return color_image, depth_image
            return None, None
        finally:
            pass

class CVCamera(Camera):
    def __init__(
            self,
            index:int,
            name:str,
            max_w_h:int
        ):
        super(CVCamera, self).__init__(index, name, max_w_h)
        self.capture = cv.VideoCapture(index)

        self.idx = 0
        for fname in os.listdir(f'sequences/realsense/new'):
            camera = fname[0:4]
            if camera != self.name:
                continue
            frame_idx = int(fname[5:10])
            if frame_idx > self.idx:
                self.idx = frame_idx + 1
        pass

    def captureNext(self):
        ret, frame = self.capture.read()
        if not ret:
            raise Exception(f'No frame returned!')
        frame = self.__resize__(frame)
        if self.record:
            self.frames.append(frame)
        return frame
    
    def save_frames(self):
        for color_frame in self.frames:
            fname_color = f'sequences/realsense/new/{self.name}F{self.idx:0>5}_color.jpg'

            color_frame = np.uint8(np.round(color_frame * 255))

            cv.imwrite(fname_color, color_frame)
            idx += 1
        self.frames = []

class CameraManager():
    def get_connected_realsense_cameras(camera_names = None, rs_devices = None, max_w_h:int = 500, dataset_directory:str = None) -> List[RealSenseCamera]:
        context = rs.context()

        if rs_devices is not None and camera_names is not None and len(rs_devices) != len(camera_names):
            raise Exception(f'Number of rs_devices ({rs_devices}) should be equal to number of cam_names ({camera_names})!')
        
        realsense_devices = [device for device in context.devices]

        if camera_names is None:
            if rs_devices is None:
                camera_names = [f'C{i:0>3}' for i in range(len(realsense_devices))]
            else:
                camera_names = [f'C{i:0>3}' for i in range(len(rs_devices))]

        rs_cameras = []
        if rs_devices is None:
            serial_numbers = []
            for device, name in zip(context.devices, camera_names):
                if device.get_info(rs.camera_info.name).lower() != 'platform camera':
                    serial_number = device.get_info(rs.camera_info.serial_number)
                    serial_numbers.append(serial_number)
        else:
            serial_numbers = rs_devices
        for serial_number, camera_name in zip(serial_numbers, camera_names):
            rs_camera = RealSenseCamera(serial_number, camera_name, max_w_h, dataset_directory)
            rs_cameras.append(rs_camera)
        return rs_cameras
    
    def get_connectec_cv_cameras(camera_names, cv_devices, max_w_h:int) -> List[CVCamera]:
        if cv_devices is not None and camera_names is not None and len(cv_devices) != len(camera_names):
            raise Exception(f'Number of cv_devices ({cv_devices}) should be equal to number of cam_names ({camera_names})!')
        
        if camera_names is None:
            if cv_devices is None:
                max_cameras = 10
                available_cv_captures = []
                for i in range(max_cameras):
                    cap = cv.VideoCapture(i, cv.CAP_DSHOW)
                    
                    if not cap.read()[0]:
                        print(f"Camera index {i:02d} not found...")
                        continue
                    
                    available_cv_captures.append(i)
                    cap.release()
                camera_names = [f'C{i:0>3}' for i in available_cv_captures]
            else:
                camera_names = [f'C{i:0>3}' for i in range(len(cv_devices))]

        cv_cameras = []
        if cv_devices is None:
            camera_ids = []
            for device_id, name in zip(available_cv_captures, camera_names):
                if device_id.get_info(rs.camera_info.name).lower() != 'platform camera':
                    camera_id = device_id.get_info(rs.camera_info.serial_number)
                    camera_ids.append(camera_id)
        else:
            camera_ids = cv_devices
        for camera_id in camera_ids:
            cv_camera = CVCamera(camera_id, name, max_w_h)
            cv_cameras.append(cv_camera)
        return cv_cameras

def show_rec(img, record, t_switch, blink_interval = 1500):
    if record:
        if datetime.now() - t_switch > timedelta(milliseconds=blink_interval // 2):
            img = cv.circle(img, (40, 40), radius=20, thickness=-1, color=(0,0,255))
        if datetime.now() - t_switch > timedelta(milliseconds=blink_interval):
            t_switch = datetime.now()
        img = cv.putText(img, 'recording', (70, 54), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=1.7, color=(0,0,255), thickness=2)
    return img, t_switch

def makedirs(path):
    if not os.path.isdir(path):
        os.makedirs(path)

def init_directories(args):
    out_dir = '' if not args.dataset_directory else args.dataset_directory + '/'
    out_dir += f'sequences/{args.cap_mode}'
    
    phases = ['train', 'val']
    for phase in phases:
        makedirs(f'{out_dir}/{phase}')
    makedirs(f'{out_dir}/new')

def capture_realsense(args):
    cameras:List[RealSenseCamera] = CameraManager.get_connected_realsense_cameras(args.cam_names, args.rs_devices, args.max_w_h, args.dataset_directory)

    framerate = args.framerate
    record = False
    i = 0
    while True:
        cycle_start = datetime.now()
        for camera in cameras:
            camera.record = record
            color_frame, depth_frame = camera.captureNext()

            fontScale = camera.max_w_h / 500
            img_color = color_frame.copy()
            img_color = cv.putText(img_color, f'{framerate:0.2f} FPS', (20, int(30 * fontScale / 0.6)), cv.FONT_HERSHEY_DUPLEX, fontScale=fontScale, color=(1, 1, 1))
            img_color = cv.putText(img_color, f'frame_cnt {len(camera.frames)}', (20, int(50 * fontScale / 0.6)), cv.FONT_HERSHEY_DUPLEX, fontScale=fontScale, color=(1, 1, 1))

            depth_frame = (depth_frame - depth_frame.min()) / (depth_frame.max() - depth_frame.min())
            depth_frame = cv.applyColorMap(np.uint8(depth_frame * 255), cv.COLORMAP_DEEPGREEN)

            img_color = np.uint8(img_color * 255)
            img_color = cv.cvtColor(img_color, cv.COLOR_BGR2RGB)

            cv.imshow(f'{camera.name} ({camera.id}) - color_frame', img_color)
            #cv.imshow(f'{camera.id} - depth_frame', depth_frame)

            if record and len(camera.frames) >= args.frame_buffer_size:
                camera.save_frames()
        
        key = cv.waitKey(1)

        while 1 / (datetime.now() - cycle_start).total_seconds() > 40 / 30 * args.framerate:
            time.sleep(1e-6)

        new_framerate = 1 / (datetime.now() - cycle_start).total_seconds()
        framerate += 1e-1 * (new_framerate - framerate)
        if framerate < (20 / 30) * args.framerate:
            raise Exception(f'Framerate dropped: {framerate} FPS')
        
        if key == ord('r'):
            record = not record
            print(f'recording: {record}')
            if not record:
                for camera in cameras:
                    camera.save_frames()
                print('Frames saved.')
        if key == ord('s'):
            record = False
            print(f'recording: {record}')
            for camera in cameras:
                camera.save_frames()
            print('Frames saved.')

        i += 1
    pass

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cap_mode', type=str, default='realsense', required=True)
    parser.add_argument('--cam_names', type=str, nargs='+', default=None)
    parser.add_argument('--rs_devices', type=str, nargs='+', default=None)
    parser.add_argument('--max_w_h', type=int, default=5000)
    parser.add_argument('--dataset_directory', type=str, required=True)
    parser.add_argument('--framerate', type=float, default=30)
    parser.add_argument('--frame_buffer_size', type=int, default=1000)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()

    init_directories(args)

    if args.cap_mode == 'realsense':
        capture_realsense(args)
    else:
        raise Exception(f'cap_mode {args.cap_mode} is not supported.')