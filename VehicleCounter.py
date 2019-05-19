import cv2
from trackers.tracker import create_blob, add_new_blobs, remove_duplicates
import numpy as np
from collections import OrderedDict
from detectors.detector import get_bounding_boxes
import uuid
import os
import contextlib
from datetime import datetime
import argparse
from utils.detection_roi import get_roi_frame, draw_roi
from counter import get_counting_line, is_passed_counting_line

class VehicleCounter():

    __init__(self,detector='yolo', tracker='kcf', droi=None, show_droi = False, mctf=3, di=10, log=False, record_path='./videos/output.avi', log_file='log.txt', cl_position='bottom'):
        self.detector = detetector
        self.droi =  droi
        self.show_droi = show_droi
        self.mctf = mctf
        self.di = di
        self.record = record
        self.record_destination = record_path
        self.log_file_name = log_file
        self.cl_position = cl_position


    def initialize(self):
        if(not self.droi):
            cap = cv2.VideoCapture(self.source)
            _, frame = cap.read()
            self.frame_height, self.frame_width, _ = frame.shape
            droi = [(0, 0), (frame_width, 0), (frame_width, frame_height), (0, frame_height)]
        else:

        self.blobs = OrderedDict()
        self.blob_id = 1
        self.frame_counter = 0
        self.vehicle_count = 0
        self.counting_line = self.get_counting_line(clposition, f_width, f_height)
    
    def initialize_recording(self):
        self.output_video = cv2.VideoWriter(self.record_destination, cv2.VideoWriter_fourcc('M','J','P','G'), 30, (self.frame_width, self.frame_height))
        log_file_name = 'log.txt'
        with contextlib.suppress(FileNotFoundError):
            os.remove(log_file_name)
        log_file = open(log_file_name, 'a')
        log_file.write('vehicle_id, count, datetime\n')
        log_file.flush()

    def initialize_blobs(self):
        droi_frame = get_roi_frame(frame, droi)
        initial_bboxes = get_bounding_boxes(droi_frame, detector)
        for box in initial_bboxes:
            _blob = create_blob(box, frame, tracker)
            blobs[blob_id] = _blob
            blob_id += 1

    def set_detector(self,detector):
        self.detector = detector

    def set_tracker(self, tracker):
        self.tracker = tracker

    def set_droi(self, droi):
        self.droi = droi

    def show_droi(self, flag)
        self.show_droi = flag

    def set_enable_log(self, flag):
        self.set_enable_recording = flag

    def set_record_destination(self,path_to_file):
        self.record_destination = path_to_file

    def set_cl_position(self, cl_position):
        self.cl_position = cl_position

    def count_vehicles(self, frame):
        for _id, blob in list(blobs.items()):
            # update trackers
            success, box = blob.tracker.update(frame)
            if success:
                blob.num_consecutive_tracking_failures = 0
                blob.update(box)
            else:
                blob.num_consecutive_tracking_failures += 1

            # delete untracked blobs
            if blob.num_consecutive_tracking_failures >= MAX_CONSECUTIVE_TRACKING_FAILURES:
                del blobs[_id]

            # count vehicles
            if is_passed_counting_line(blob.centroid, counting_line, clposition) and not blob.counted:
                blob.counted = True
                vehicle_count += 1

                # log count data to a file (vehicle_id, count, datetime)
                if args.record:
                    _row = '{0}, {1}, {2}\n'.format('v_' + str(_id), vehicle_count, datetime.now())
                    log_file.write(_row)
                    log_file.flush()

        if frame_counter >= DETECTION_INTERVAL:
            # rerun detection
            droi_frame = get_roi_frame(frame, droi)
            boxes = get_bounding_boxes(droi_frame, detector)
            blobs, current_blob_id = add_new_blobs(boxes, blobs, frame, tracker, blob_id, counting_line, clposition)
            blob_id = current_blob_id
            blobs = remove_duplicates(blobs)
            frame_counter = 0

        # draw and label blob bounding boxes
        for _id, blob in blobs.items():
            (x, y, w, h) = [int(v) for v in blob.bounding_box]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, 'v_' + str(_id), (x, y - 2), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # draw counting line
        cv2.line(frame, counting_line[0], counting_line[1], (0, 255, 0), 3)

        # display vehicle count
        cv2.putText(frame, 'Count: ' + str(vehicle_count), (20, 60), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 0, 0), 2, cv2.LINE_AA)

        # show detection roi
        if self.show_droi:
            frame = draw_roi(frame, droi)

        # save frame in video output
        if self.record:
            selfoutput_video.write(frame)

        # visualize vehicle counting
        resized_frame = cv2.resize(frame, (858, 480))
        frame_counter += 1
        return resized_frame

    else:
        print('End of video.')
        # end video loop if on the last frame
        break



if __name__ == '__main__':
