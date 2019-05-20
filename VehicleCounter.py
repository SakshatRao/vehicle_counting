import cv2
from vehicle_counting.trackers.tracker import create_blob, add_new_blobs, remove_duplicates
import numpy as np
from collections import OrderedDict
from .detectors.detector import get_bounding_boxes
import uuid
import os
import contextlib
from datetime import datetime
import argparse
from vehicle_counting.utils.detection_roi import get_roi_frame, draw_roi
from vehicle_counting.counter import get_counting_line, is_passed_counting_line

class VehicleCounter():

    def __init__(self, detector='yolo', tracker='kcf', droi=None, show_droi = False, mctf=3, di=10, record=False, record_path='./videos/output.avi', log_file='log.txt', cl_position='bottom'):
        self.detector = detector
        self.tracker = tracker
        self.droi =  droi
        self.show_droi = show_droi
        self.mctf = mctf
        self.detection_interval = di
        self.record = record
        self.record_destination = record_path
        self.log_file_name = log_file
        self.cl_position = cl_position
        self.is_initialized = False

    def initialize(self):
        if(not self.droi):
            self.frame_height, self.frame_width, _ = self.frame.shape
            self.droi = [(0, 0), (self.frame_width, 0), (self.frame_width, self.frame_height), (0, self.frame_height)]

        self.blobs = OrderedDict()
        self.blob_id = 1
        self.frame_counter = 0
        self.vehicle_count = 0
        self.counting_line = get_counting_line(self.cl_position, self.frame_width, self.frame_height)

    def reset_counter():
        self.frame = None
        self.droi = None
        self.is_initialized = False
    
    def initialize_recording(self):
        self.output_video = cv2.VideoWriter(self.record_destination, cv2.VideoWriter_fourcc('M','J','P','G'), 30, (self.frame_width, self.frame_height))
        log_file_name = 'log.txt'
        with contextlib.suppress(FileNotFoundError):
            os.remove(log_file_name)
        log_file = open(log_file_name, 'a')
        log_file.write('vehicle_id, count, datetime\n')
        log_file.flush()

    def initialize_blobs(self):
        droi_frame = get_roi_frame(self.frame, self.droi)
        initial_bboxes = get_bounding_boxes(droi_frame, self.detector)
        for box in initial_bboxes:
            _blob = create_blob(box, frame, self.tracker)
            self.blobs[self.blob_id] = _blob
            self.blob_id += 1

    def set_detector(self,detector):
        self.detector = detector

    def set_tracker(self, tracker):
        self.tracker = tracker

    def set_droi(self, droi):
        self.droi = droi

    def show_droi(self, flag):
        self.show_droi = flag

    def set_enable_log(self, flag):
        self.set_enable_recording = flag

    def set_record_destination(self,path_to_file):
        self.record_destination = path_to_file

    def set_cl_position(self, cl_position):
        self.cl_position = cl_position

    def count_vehicles(self, frame):
        self.frame = frame
        if(not self.is_initialized):
            self.initialize()
            self.is_initialized = True

        self.initialize_recording
        self.initialize_blobs
                
        for _id, blob in list(self.blobs.items()):
            # update trackers
            success, box = blob.tracker.update(self.frame)
            if success:
                blob.num_consecutive_tracking_failures = 0
                blob.update(box)
            else:
                blob.num_consecutive_tracking_failures += 1

            # delete untracked blobs
            if blob.num_consecutive_tracking_failures >= self.mctf:
                del self.blobs[_id]

            # count vehicles
            if is_passed_counting_line(blob.centroid, self.counting_line, self.cl_position) and not blob.counted:
                blob.counted = True
                self.vehicle_count += 1

                # log count data to a file (vehicle_id, count, datetime)
                if self.record:
                    _row = '{0}, {1}, {2}\n'.format('v_' + str(_id), self.vehicle_count, datetime.now())
                    log_file.write(_row)
                    log_file.flush()

        if self.frame_counter >= self.detection_interval:
            # rerun detection
            droi_frame = get_roi_frame(self.frame, self.droi)
            boxes = get_bounding_boxes(droi_frame, self.detector)
            self.blobs, current_blob_id = add_new_blobs(boxes, self.blobs, self.frame, self.tracker, self.blob_id, self.counting_line, self.cl_position)
            self.blob_id = current_blob_id
            self.blobs = remove_duplicates(self.blobs)
            self.frame_counter = 0

        # draw and label blob bounding boxes
        for _id, blob in self.blobs.items():
            (x, y, w, h) = [int(v) for v in blob.bounding_box]
            cv2.rectangle(self.frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(self.frame, 'v_' + str(_id), (x, y - 2), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # draw counting line
        cv2.line(self.frame, self.counting_line[0], self.counting_line[1], (0, 255, 0), 3)

        # display vehicle count
        cv2.putText(self.frame, 'Count: ' + str(self.vehicle_count), (20, 60), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 0, 0), 2, cv2.LINE_AA)

        # show detection roi
        if self.show_droi:
            self.frame = draw_roi(self.frame, self.droi)

        # save frame in video output
        if self.record:
            self.output_video.write(self.frame)

        # visualize vehicle counting
        resized_frame = cv2.resize(self.frame, (858, 480))
        self.frame_counter += 1
        return resized_frame


if __name__ == '__main__':
    pass
