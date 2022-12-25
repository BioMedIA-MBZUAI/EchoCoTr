"""EchoNet Dataset."""

import os
import collections
import pandas
import cv2

import numpy as np
import skimage.draw
import torchvision

class EchoNet(torchvision.datasets.VisionDataset):
    def __init__(self, root=None,
                 split="train",
                 mean=0.,
                 std=1.,
                 frames=16,
                 frequency=2,
                 max_frames=250,
                 pad=None):

        assert(root is not None)

        super().__init__(root)

        self.split = split
        self.mean = mean
        self.std = std
        self.frames = frames
        self.max_frames = max_frames
        self.frequency = frequency
        self.pad = pad

        self.vnames, self.outcome = [], []
        self.read_filelist()

        self.frames_list = collections.defaultdict(list)
        self.trace = collections.defaultdict(_defaultdict_of_lists)

        self.read_volumetracings()

        self.filter_videos()

        print("{} dataset size: {}".format(split, len(self.vnames)))

    def read_filelist(self):
        with open(os.path.join(self.root, "FileList.csv")) as f:
            self.file_header = f.readline().strip().split(",")
            filename_index = self.file_header.index("FileName")
            split_index = self.file_header.index("Split")

            for line in f:
                line_split = line.strip().split(',')

                filename = os.path.splitext(line_split[filename_index])[0] + ".avi"
                file_split = line_split[split_index].lower()

                if self.split in ["all", file_split] and os.path.exists(os.path.join(self.root, "Videos", filename)):
                    self.vnames.append(filename)
                    self.outcome.append(line_split)

        self.check_missing_videos()


    def check_missing_videos(self):
        missing_videos = set(self.vnames) - set(os.listdir(os.path.join(self.root, "Videos")))
        if len(missing_videos) != 0:
            print("{} videos are missing in {}:".format(len(missing_videos), os.path.join(self.root, "Videos")))
            for f in sorted(missing):
                print("\t", f)
            raise FileNotFoundError(os.path.join(self.root, "Videos", sorted(missing_videos)[0]))

    def read_volumetracings(self):
        with open(os.path.join(self.root, "VolumeTracings.csv")) as f:
            header = f.readline().strip().split(",")
            assert header == ["FileName", "X1", "Y1", "X2", "Y2", "Frame"]

            for line in f:
                filename, x1, y1, x2, y2, frame = line.strip().split(',')
                x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
                frame = int(frame)
                if frame not in self.trace[filename]:
                    self.frames_list[filename].append(frame)
                self.trace[filename][frame].append((x1, y1, x2, y2))

        for filename in self.frames_list:
            for frame in self.frames_list[filename]:
                 self.trace[filename][frame] = np.array(self.trace[filename][frame])


    def filter_videos(self):
        min_frames = 2
        videos_to_keep = [len(self.frames_list[f]) >= min_frames for f in self.vnames]
        self.vnames = [f for (f, k) in zip(self.vnames, videos_to_keep) if k]
        self.outcome = [f for (f, k) in zip(self.outcome, videos_to_keep) if k]

    def __getitem__(self, index):
        video = os.path.join(self.root, "Videos", self.vnames[index])

        video = self.load_video(video).astype(np.float32)

        video = self.normalize_video(video)

        video = self.sample_video(video)

        if self.pad is not None:
            video = self.pad_video(video)

        ef = np.float32(self.outcome[index][self.file_header.index("EF")])

        return video, ef

    def __len__(self):
        return len(self.vnames)

    def normalize_video(self, video):
        if isinstance(self.mean, (float, int)):
            video -= self.mean
        else:
            video -= self.mean.reshape(3, 1, 1, 1)

        if isinstance(self.std, (float, int)):
            video /= self.std
        else:
            video /= self.std.reshape(3, 1, 1, 1)

        return video

    def sample_video(self, video):
        c, f, h, w = video.shape
        frames = self.frames
        frames = min(frames, self.max_frames)

        if f < frames * self.frequency:
            video = np.concatenate((video, np.zeros((c, frames * self.frequency - f, h, w), video.dtype)), axis=1)
            c, f, h, w = video.shape  # pylint: disable=E0633

        start = np.random.choice(f - (frames - 1) * self.frequency, 1)

        video = tuple(video[:, s + self.frequency * np.arange(frames), :, :] for s in start)[0]

        return video


    def pad_video(self, video):
        if self.pad is None:
            return video

        c, l, h, w = video.shape
        tvideo = np.zeros((c, l, h + 2 * self.pad, w + 2 * self.pad), dtype=video.dtype)
        tvideo[:, :, self.pad:-self.pad, self.pad:-self.pad] = video  # pylint: disable=E1130
        i, j = np.random.randint(0, 2 * self.pad, 2)

        return tvideo[:, :, i:(i + h), j:(j + w)]

    def load_video(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(path)

        capture = cv2.VideoCapture(path)

        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        video = np.zeros((count, height, width, 3), np.uint8)

        for i in range(count):
            out, frame = capture.read()
            if not out:
                raise ValueError("Problem when reading frame #{} of {}.".format(i, filename))

            video[i, :, :] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        return video.transpose((3, 0, 1, 2))

def _defaultdict_of_lists():
    return collections.defaultdict(list)
