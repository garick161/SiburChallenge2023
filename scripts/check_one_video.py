from show_results_model import get_frames_with_bb


if __name__ == '__main__':
    weights_path = 'weights_ver5.pt'
    get_frames_with_bb(video_path='../prepair_dataset/train/bridge_down/790f22429c3b917a.mp4',
                       path_to_dir='../problem_frames',
                       path_to_weights=weights_path)