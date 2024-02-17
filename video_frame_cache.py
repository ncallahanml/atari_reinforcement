import os
import joblib
import gc
import cv2

class VideoFrameCache():
    def __init__(self, cache_dir='test_files/frame_cache', width=720, height=1280, channel=3, dtype=np.uint8):
        if not os.path.exists(cache_dir):
            raise ValueError('Cache directory does not exist')
        self.cache_dir = cache_dir
        self.cache = list()
        self.exp_shape = (width, height, channel)
        self.dtype = dtype
        return
    
    def cache_append(self, frame, delete=True):
        assert frame.shape == self.exp_shape, f'Frame invalid shape, expected {self.exp_shape}, got {frame.shape}'
        assert frame.dtype == self.dtype
        cache_path = os.path.join(self.cache_dir, f'{len(self.cache)}.jpg')
        self.cache.extend(joblib.dump(frame[:,:,::-1], cache_path))
        if delete:
            del frame
        return
        
    def finish(self, vid_path, fps=6, print_=True):
        assert vid_path.endswith('.mp4'), 'Extension must be mp4'
        gc.collect()
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        vw = cv2.VideoWriter(vid_path, fourcc, float(fps), (self.exp_shape[1], self.exp_shape[0]))
        for frame_path in self.cache:
            frame = joblib.load(frame_path)
            vw.write(frame)
        vw.release()
        if print_:
            print(f'Video of length {len(self.cache)} saved successfully at {vid_path}')
        return